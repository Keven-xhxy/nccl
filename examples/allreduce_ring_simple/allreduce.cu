// One named kernel calling NCCL's original runRing and Simple primitives.
#include <cuda_runtime.h>
#define NCCL_PRIMITIVES_STANDALONE 1
#include "common.h"
#include "all_reduce.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

constexpr int kThreads = 128;
constexpr int kStepElements = 1024;
constexpr int kChunkElements = 1024;
constexpr int kChunkSteps = ALLREDUCE_CHUNKSTEPS;
constexpr size_t kGuard = 16;
constexpr float kSentinel = -123456.0f;
using Simple = ProtoSimple<ALLREDUCE_CHUNKSTEPS / ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS>;

// A real NCCL_STEPS-slot Simple FIFO for one incoming ring edge.
struct Edge {
  float data[NCCL_STEPS * kStepElements];
  uint64_t head;
  uint64_t tail;
};

static_assert(kChunkElements * sizeof(float) % ncclProtoGrainSize(NCCL_PROTO_SIMPLE) == 0,
              "The initial chunk size must be an integer number of Simple grains");

// The entry point only initializes shared context and calls the original template.
__global__ void allreduce_ring_simple(ncclKernelComm comm, ncclDevChannel channel,
                                      ncclDevWorkColl work) {
  const int tid = threadIdx.x;
  if (comm.nRanks == 1) {
    const float* input = static_cast<const float*>(work.sendbuff);
    float* output = static_cast<float*>(work.recvbuff);
    for (size_t i = tid; i < work.cbd.countLo; i += blockDim.x) output[i] = input[i];
    return;
  }
  __shared__ ncclDevWorkColl ringWork;
  if (tid == 0) {
    ncclShmem.comm = comm;
    ncclShmem.channel = channel;
    ncclShmem.channelId = 0;
    ncclShmem.aborted = 0;
    ringWork = work;
  }
  __syncthreads();
  runRing<float, FuncSum<float>, Simple>(tid, blockDim.x, &ringWork);
}

#define CUDA(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
  std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(e)); \
  std::exit(1); } } while (0)

struct Rank {
  cudaStream_t stream;
  Edge* edge;
  ncclDevChannelPeer* peers;
  ncclDevChannelPeer** peerPointers;
  uint32_t* abortFlag;
  ncclKernelComm comm = {};
  ncclDevChannel channel = {};
  size_t scratchBytes = 0;
  uint64_t expectedStep = 0;
  float* input;
  float* output;
};

static float input_value(int rank, size_t index) {
  return float(int((size_t(rank) * 5 + index * 3) % 17) - 8);
}

static bool run_case(std::vector<Rank>& ranks, size_t count, bool in_place) {
  int n = int(ranks.size());
  std::vector<std::vector<float>> inputs(n, std::vector<float>(count + kGuard, kSentinel));
  std::vector<float> initial(count + kGuard, kSentinel), actual(count + kGuard);
  std::vector<float> expected(count, 0.0f);
  for (int r = 0; r < n; r++) {
    CUDA(cudaSetDevice(r));
    for (size_t i = 0; i < count; i++) {
      inputs[r][i] = input_value(r, i);
      expected[i] += inputs[r][i];
    }
    // All input buffers remain alive until every asynchronous copy completes.
    CUDA(cudaMemcpyAsync(ranks[r].input, inputs[r].data(), inputs[r].size() * sizeof(float),
                         cudaMemcpyHostToDevice, ranks[r].stream));
    CUDA(cudaMemcpyAsync(ranks[r].output, initial.data(), initial.size() * sizeof(float),
                         cudaMemcpyHostToDevice, ranks[r].stream));
  }
  // Finish every peer's initialization/input copies before starting the ring.
  // FIFO counters and connection steps deliberately persist across test cases.
  for (int r = 0; r < n; r++) {
    CUDA(cudaSetDevice(r));
    CUDA(cudaStreamSynchronize(ranks[r].stream));
  }
  // Launch every GPU before waiting on any stream: ring peers must run together.
  for (int r = 0; r < n; r++) {
    CUDA(cudaSetDevice(r));
    float* dst = in_place ? ranks[r].input : ranks[r].output;
    // One channel: all elements belong to countLo, with an initial 1024-float chunk.
    ncclDevWorkColl work = {};
    work.channelLo = work.channelHi = 0;
    work.nWarps = kThreads / WARP_SIZE;
    work.oneNode = 1;
    work.sendbuff = ranks[r].input;
    work.recvbuff = dst;
    work.cbd.countLo = count;
    work.cbd.chunkGrainsLo = kChunkElements * sizeof(float) / ncclProtoGrainSize(NCCL_PROTO_SIMPLE);
    allreduce_ring_simple<<<1, kThreads, ranks[r].scratchBytes, ranks[r].stream>>>(
        ranks[r].comm, ranks[r].channel, work);
    CUDA(cudaGetLastError());
  }
  bool passed = true;
  for (int r = 0; r < n; r++) {
    CUDA(cudaSetDevice(r));
    CUDA(cudaStreamSynchronize(ranks[r].stream));
    const float* dst = in_place ? ranks[r].input : ranks[r].output;
    CUDA(cudaMemcpy(actual.data(), dst, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
    size_t tiles = (count + size_t(n) * kChunkElements - 1) / (size_t(n) * kChunkElements);
    ranks[r].expectedStep += n == 1 ? 0 : uint64_t(tiles) * 2 * (n - 1) * kChunkSteps;
    if (n > 1) {
      uint64_t steps[2];
      CUDA(cudaMemcpy(steps, &ranks[r].edge->head, sizeof(steps), cudaMemcpyDeviceToHost));
      ncclDevChannelPeer recvPeer, sendPeer;
      CUDA(cudaMemcpy(&recvPeer, ranks[r].peers + (r + n - 1) % n, sizeof(recvPeer), cudaMemcpyDeviceToHost));
      CUDA(cudaMemcpy(&sendPeer, ranks[r].peers + (r + 1) % n, sizeof(sendPeer), cudaMemcpyDeviceToHost));
      if (steps[0] != ranks[r].expectedStep || steps[1] != ranks[r].expectedStep ||
          recvPeer.recv[0].step != ranks[r].expectedStep || sendPeer.send[0].step != ranks[r].expectedStep) {
        std::fprintf(stderr, "FAIL rank=%d FIFO/connection step persistence (expected %llu)\n",
                     r, (unsigned long long)ranks[r].expectedStep);
        passed = false;
      }
    }
    for (size_t i = 0; i < actual.size(); i++) {
      float want = i < count ? expected[i] : kSentinel;
      if (!std::isfinite(actual[i]) || actual[i] != want) {
        std::fprintf(stderr, "FAIL rank=%d index=%zu actual=%g expected=%g\n", r, i, actual[i], want);
        passed = false;
        break;
      }
    }
  }
  std::printf("%s ranks=%d count=%zu inplace=%d\n", passed ? "PASS" : "FAIL", n, count, in_place);
  return passed;
}

int main(int argc, char** argv) {
  int n = 2;
  size_t count = 4097;
  bool self_test = false, in_place = false;
  for (int i = 1; i < argc; i++) {
    if (!std::strcmp(argv[i], "--self-test")) self_test = true;
    else if (!std::strcmp(argv[i], "--in-place")) in_place = true;
    else if ((!std::strcmp(argv[i], "--ranks") || !std::strcmp(argv[i], "--count")) && i + 1 < argc) {
      bool is_rank = !std::strcmp(argv[i], "--ranks");
      char* end;
      const char* text = argv[++i];
      unsigned long long value = std::strtoull(text, &end, 10);
      if (text[0] == '-' || end == text || *end || value > (is_rank ? 8ULL : 16777216ULL) || (is_rank && !value)) {
        std::fprintf(stderr, "Invalid numeric argument: %s\n", text);
        return 2;
      }
      if (is_rank) n = int(value); else count = size_t(value);
    } else {
      std::fprintf(stderr, "Usage: %s [--ranks 1..8] [--count 0..16777216] [--in-place] [--self-test]\n", argv[0]);
      return 2;
    }
  }
  int available = 0;
  cudaError_t discovery = cudaGetDeviceCount(&available);
  if (discovery != cudaSuccess || available < n) {
    std::fprintf(stderr, "SKIP: requires %d CUDA GPUs (found %d, %s)\n", n, available, cudaGetErrorString(discovery));
    return 77;
  }
  for (int r = 0; r < n && n > 1; r++) {
    int next = (r + 1) % n, accessible = 0;
    CUDA(cudaDeviceCanAccessPeer(&accessible, r, next));
    if (!accessible) {
      std::fprintf(stderr, "SKIP: ring edge %d -> %d requires CUDA P2P\n", r, next);
      return 77;
    }
  }
  std::vector<size_t> sizes = self_test ? std::vector<size_t>{0, 1, 17, 1023, 1024, 1025, 4097, 65539, 1048576}
                                      : std::vector<size_t>{count};
  size_t capacity = *std::max_element(sizes.begin(), sizes.end()) + kGuard;
  std::vector<Rank> ranks(n);
  for (int r = 0; r < n; r++) {
    CUDA(cudaSetDevice(r));
    if (n > 1) {
      cudaError_t enabled = cudaDeviceEnablePeerAccess((r + 1) % n, 0);
      if (enabled == cudaErrorPeerAccessAlreadyEnabled) (void)cudaGetLastError();
      else CUDA(enabled);
    }
    CUDA(cudaStreamCreateWithFlags(&ranks[r].stream, cudaStreamNonBlocking));
    CUDA(cudaMalloc(&ranks[r].edge, sizeof(Edge)));
    CUDA(cudaMemsetAsync(ranks[r].edge, 0, sizeof(Edge), ranks[r].stream));
    CUDA(cudaMalloc(&ranks[r].peers, n * sizeof(ncclDevChannelPeer)));
    CUDA(cudaMalloc(&ranks[r].peerPointers, n * sizeof(ncclDevChannelPeer*)));
    CUDA(cudaMalloc(&ranks[r].abortFlag, sizeof(uint32_t)));
    CUDA(cudaMemsetAsync(ranks[r].abortFlag, 0, sizeof(uint32_t), ranks[r].stream));
    cudaDeviceProp prop;
    CUDA(cudaGetDeviceProperties(&prop, r));
    ranks[r].scratchBytes = ncclShmemScratchWarpSize(prop.major * 100 + prop.minor * 10) * (kThreads / WARP_SIZE);
    CUDA(cudaMalloc(&ranks[r].input, capacity * sizeof(float)));
    CUDA(cudaMalloc(&ranks[r].output, capacity * sizeof(float)));
  }
  // Hand-wire the two ncclConnInfo records used by the original primitives.
  for (int r = 0; r < n; r++) {
    CUDA(cudaSetDevice(r));
    int prev = (r + n - 1) % n, next = (r + 1) % n;
    std::vector<ncclDevChannelPeer> peers(n);
    std::vector<ncclDevChannelPeer*> pointers(n);
    auto connection = [](Edge* edge) {
      ncclConnInfo conn = {};
      conn.buffs[NCCL_PROTO_SIMPLE] = reinterpret_cast<char*>(edge->data);
      conn.head = &edge->head;
      conn.tail = &edge->tail;
      conn.stepSize = kStepElements * sizeof(float);
      return conn;
    };
    peers[prev].recv[0] = connection(ranks[r].edge);
    peers[next].send[0] = connection(ranks[next].edge);
    for (int peer = 0; peer < n; peer++) pointers[peer] = ranks[r].peers + peer;
    CUDA(cudaMemcpy(ranks[r].peers, peers.data(), n * sizeof(ncclDevChannelPeer), cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(ranks[r].peerPointers, pointers.data(), n * sizeof(ncclDevChannelPeer*), cudaMemcpyHostToDevice));
    ranks[r].comm.rank = r;
    ranks[r].comm.nRanks = n;
    ranks[r].comm.nNodes = 1;
    ranks[r].comm.buffSizes[NCCL_PROTO_SIMPLE] = NCCL_STEPS * kStepElements * sizeof(float);
    ranks[r].comm.abortFlag = ranks[r].abortFlag;
    ranks[r].channel.peers = ranks[r].peerPointers;
    ranks[r].channel.ring.prev = prev;
    ranks[r].channel.ring.next = next;
    ranks[r].channel.ring.index = r;
  }
  bool passed = true;
  for (size_t size : sizes) {
    for (int repeat = 0; repeat < (self_test ? 2 : 1); repeat++) {
      if (self_test) {
        passed &= run_case(ranks, size, false);
        passed &= run_case(ranks, size, true);
      } else passed &= run_case(ranks, size, in_place);
    }
  }
  for (int r = 0; r < n; r++) {
    CUDA(cudaSetDevice(r));
    CUDA(cudaFree(ranks[r].input));
    CUDA(cudaFree(ranks[r].output));
    CUDA(cudaFree(ranks[r].edge));
    CUDA(cudaFree(ranks[r].peers));
    CUDA(cudaFree(ranks[r].peerPointers));
    CUDA(cudaFree(ranks[r].abortFlag));
    CUDA(cudaStreamDestroy(ranks[r].stream));
  }
  return passed ? 0 : 1;
}

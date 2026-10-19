// Compile one operation at a time: one translation unit and one named kernel.
#include <cuda_runtime.h>
#define NCCL_PRIMITIVES_STANDALONE 1
#include "common.h"

#if defined(EXAMPLE_BROADCAST)
#include "broadcast.h"
using Simple = ProtoSimple<BROADCAST_CHUNKSTEPS / BROADCAST_SLICESTEPS, BROADCAST_SLICESTEPS>;
#define EXAMPLE_KERNEL broadcast_ring_simple
constexpr int kChunkSteps = BROADCAST_CHUNKSTEPS;
#elif defined(EXAMPLE_REDUCE)
#include "reduce.h"
using Simple = ProtoSimple<REDUCE_CHUNKSTEPS / REDUCE_SLICESTEPS, REDUCE_SLICESTEPS>;
#define EXAMPLE_KERNEL reduce_ring_simple
constexpr int kChunkSteps = REDUCE_CHUNKSTEPS;
#elif defined(EXAMPLE_ALLGATHER)
#include "all_gather.h"
using Simple = ProtoSimple<ALLGATHER_CHUNKSTEPS / ALLGATHER_SLICESTEPS, ALLGATHER_SLICESTEPS>;
#define EXAMPLE_KERNEL allgather_ring_simple
constexpr int kChunkSteps = ALLGATHER_CHUNKSTEPS;
#elif defined(EXAMPLE_REDUCESCATTER)
#include "reduce_scatter.h"
using Simple = ProtoSimple<REDUCESCATTER_CHUNKSTEPS / REDUCESCATTER_SLICESTEPS, REDUCESCATTER_SLICESTEPS>;
#define EXAMPLE_KERNEL reducescatter_ring_simple
constexpr int kChunkSteps = REDUCESCATTER_CHUNKSTEPS;
#else
#include "sendrecv.h"
#define EXAMPLE_P2P 1
#if defined(EXAMPLE_SENDRECV)
#define EXAMPLE_KERNEL sendrecv_simple
#elif defined(EXAMPLE_ALLTOALL)
#define EXAMPLE_KERNEL alltoall_simple
#elif defined(EXAMPLE_GATHER)
#define EXAMPLE_KERNEL gather_simple
#elif defined(EXAMPLE_SCATTER)
#define EXAMPLE_KERNEL scatter_simple
#else
#error Select an EXAMPLE operation through the Makefile
#endif
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define STRINGIFY_(x) #x
#define STRINGIFY(x) STRINGIFY_(x)
constexpr const char* kName = STRINGIFY(EXAMPLE_KERNEL);
constexpr int kGroupThreads = 128;
constexpr int kStepElements = 1024;
constexpr int kChunkElements = 1024;
constexpr int kChunkBytes = kChunkElements * sizeof(float);
constexpr size_t kGuard = 16;
constexpr float kSentinel = -123456.0f;
#if defined(EXAMPLE_P2P)
constexpr int kThreads = 2 * kGroupThreads;
constexpr int kConnIndex = 1;
#else
constexpr int kThreads = kGroupThreads;
constexpr int kConnIndex = 0;
#endif
#if defined(EXAMPLE_BROADCAST) || defined(EXAMPLE_REDUCE) || defined(EXAMPLE_GATHER) || defined(EXAMPLE_SCATTER)
constexpr bool kRooted = true;
#else
constexpr bool kRooted = false;
#endif
#if defined(EXAMPLE_SENDRECV) || defined(EXAMPLE_ALLTOALL)
constexpr bool kSupportsInPlace = false;
#else
constexpr bool kSupportsInPlace = true;
#endif

struct Edge {
  float data[NCCL_STEPS * kStepElements];
  uint64_t head, tail;
};

#if !defined(EXAMPLE_P2P)
__global__ void EXAMPLE_KERNEL(ncclKernelComm comm, ncclDevChannel channel, ncclDevWorkColl work) {
  int tid = threadIdx.x;
  if (comm.nRanks == 1) {
    auto src = static_cast<const float*>(work.sendbuff);
    auto dst = static_cast<float*>(work.recvbuff);
    for (size_t i = tid; i < work.cbd.countLo; i += blockDim.x) dst[i] = src[i];
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
#else
// A single matched transfer or a simultaneous send/recv pair. No batch dispatcher.
// AlltoAll/Gather/Scatter use this same kernel body with explicit host scheduling.
__global__ void EXAMPLE_KERNEL(ncclKernelComm comm, ncclDevChannel channel, ncclDevWorkP2p work) {
  if (work.sendRank < 0 && work.recvRank < 0) return;
  int tid = threadIdx.x;
  if (work.sendRank == comm.rank && work.recvRank == comm.rank) {
    auto src = static_cast<const float*>(work.sendAddr);
    auto dst = static_cast<float*>(work.recvAddr);
    for (size_t i = tid; i < work.sendBytes / sizeof(float); i += blockDim.x) dst[i] = src[i];
    return;
  }
  __shared__ ncclDevWorkP2p p2pWork;
  if (tid == 0) {
    ncclShmem.comm = comm;
    ncclShmem.channel = channel;
    ncclShmem.channelId = 0;
    ncclShmem.aborted = 0;
    p2pWork = work;
  }
  __syncthreads();
  if (tid < kGroupThreads) {
    if (p2pWork.sendRank >= 0)
      runSend<int8_t, FuncSum<int8_t>, ProtoSimple<1, 1>>(tid, kGroupThreads, 0, &p2pWork);
  } else {
    // Send's 128 threads use barrier IDs 15/14; recv uses 13 (group=2).
    if (p2pWork.recvRank >= 0)
      runRecv<int8_t, FuncSum<int8_t>, ProtoSimple<1, 1>>(tid - kGroupThreads, kGroupThreads, 2, &p2pWork);
  }
}
#endif

#define CUDA(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
  std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(e)); \
  std::exit(1); } } while (0)

struct Rank {
  cudaStream_t stream;
  Edge* edges;
  ncclDevChannelPeer* peers;
  ncclDevChannelPeer** peerPointers;
  int* ringRanks;
  uint32_t* abortFlag;
  float *input, *output;
  ncclKernelComm comm = {};
  ncclDevChannel channel = {};
  size_t scratchBytes;
  std::vector<uint64_t> expectedRecv, expectedSend;
};

static float input_value(int rank, size_t index) {
  return float(int((size_t(rank) * 5 + index * 3) % 17) - 8);
}

static size_t send_elements(size_t count, int n) {
#if defined(EXAMPLE_REDUCESCATTER) || defined(EXAMPLE_SCATTER) || defined(EXAMPLE_ALLTOALL)
  return count * n;
#else
  return count;
#endif
}
static size_t recv_elements(size_t count, int n) {
#if defined(EXAMPLE_ALLGATHER) || defined(EXAMPLE_GATHER) || defined(EXAMPLE_ALLTOALL)
  return count * n;
#else
  return count;
#endif
}

// Host and CPU reference share only buffer layout, not the communication schedule.
struct Layout {
  size_t sendOffset = 0, recvOffset = 0;
  bool sendOnOutput = false, recvOnInput = false;
};
static Layout layout(int rank, int root, size_t count, bool inPlace) {
  Layout x;
  if (!inPlace) return x;
#if defined(EXAMPLE_ALLGATHER)
  x.sendOnOutput = true;
  x.sendOffset = rank * count;
#elif defined(EXAMPLE_GATHER)
  x.sendOnOutput = rank == root;
  x.sendOffset = rank == root ? root * count : 0;
#elif defined(EXAMPLE_REDUCESCATTER)
  x.recvOnInput = true;
  x.recvOffset = rank * count;
#elif defined(EXAMPLE_SCATTER)
  x.recvOnInput = rank == root;
  x.recvOffset = rank == root ? root * count : 0;
#elif defined(EXAMPLE_BROADCAST) || defined(EXAMPLE_REDUCE)
  x.recvOnInput = true;
#endif
  return x;
}

static bool check_buffer(const char* buffer, int rank, const std::vector<float>& actual,
                         const std::vector<float>& expected) {
  for (size_t i = 0; i < actual.size(); i++) {
    if (!std::isfinite(actual[i]) || actual[i] != expected[i]) {
      std::fprintf(stderr, "FAIL %s rank=%d %s index=%zu actual=%g expected=%g\n",
                   kName, rank, buffer, i, actual[i], expected[i]);
      return false;
    }
  }
  return true;
}

// Number of FIFO steps sent on the directed edge src -> dst in this call.
static uint64_t edge_delta(int src, int dst, int n, int root, size_t count) {
  if (src == dst || n == 1) return 0;
#if defined(EXAMPLE_P2P)
  uint64_t chunks = std::max<size_t>(1, (count + kChunkElements - 1) / kChunkElements);
#if defined(EXAMPLE_SENDRECV)
  return dst == (src + 1) % n ? chunks : 0;
#elif defined(EXAMPLE_GATHER)
  return dst == root ? chunks : 0;
#elif defined(EXAMPLE_SCATTER)
  return src == root ? chunks : 0;
#else
  return chunks;
#endif
#else
  if (dst != (src + 1) % n) return 0;
  uint64_t chunks = (count + kChunkElements - 1) / kChunkElements;
#if defined(EXAMPLE_BROADCAST)
  return dst == root ? 0 : chunks * kChunkSteps;
#elif defined(EXAMPLE_REDUCE)
  return src == root ? 0 : chunks * kChunkSteps;
#else
  return chunks * (n - 1) * kChunkSteps;
#endif
#endif
}

static bool run_case(std::vector<Rank>& ranks, size_t count, bool inPlace, int root) {
  int n = int(ranks.size());
  size_t inSize = send_elements(count, n) + 2 * kGuard;
  size_t outSize = recv_elements(count, n) + 2 * kGuard;
  std::vector<std::vector<float>> inputs(n, std::vector<float>(inSize, kSentinel));
  std::vector<std::vector<float>> outputs(n, std::vector<float>(outSize, kSentinel));
  for (int r = 0; r < n; r++) {
    for (size_t i = 0; i < send_elements(count, n); i++) inputs[r][kGuard + i] = input_value(r, i);
    Layout x = layout(r, root, count, inPlace);
    if (x.sendOnOutput)
      for (size_t i = 0; i < count; i++) outputs[r][kGuard + x.sendOffset + i] = input_value(r, i);
    CUDA(cudaSetDevice(r));
    CUDA(cudaMemcpyAsync(ranks[r].input, inputs[r].data(), inSize * sizeof(float), cudaMemcpyHostToDevice, ranks[r].stream));
    CUDA(cudaMemcpyAsync(ranks[r].output, outputs[r].data(), outSize * sizeof(float), cudaMemcpyHostToDevice, ranks[r].stream));
  }
  // All ranks' inputs and FIFOs must be ready before any GPU enters a polling loop.
  for (int r = 0; r < n; r++) {
    CUDA(cudaSetDevice(r));
    CUDA(cudaStreamSynchronize(ranks[r].stream));
  }
  int rounds = 1;
#if defined(EXAMPLE_ALLTOALL) || defined(EXAMPLE_GATHER) || defined(EXAMPLE_SCATTER)
  rounds = n;
#endif
  for (int round = 0; round < rounds; round++) {
    // Enqueue every participating GPU before waiting for completion.
    for (int r = 0; r < n; r++) {
      CUDA(cudaSetDevice(r));
      Layout x = layout(r, root, count, inPlace);
      float* src = (x.sendOnOutput ? ranks[r].output : ranks[r].input) + kGuard + x.sendOffset;
      float* dst = (x.recvOnInput ? ranks[r].input : ranks[r].output) + kGuard + x.recvOffset;
#if !defined(EXAMPLE_P2P)
      ncclDevWorkColl work = {};
      work.channelLo = work.channelHi = 0;
      work.nWarps = kThreads / WARP_SIZE;
      work.oneNode = 1;
      work.root = root;
      work.sendbuff = src;
      work.recvbuff = dst;
      work.cbd.countLo = count;
      work.cbd.chunkGrainsLo = kChunkBytes / ncclProtoGrainSize(NCCL_PROTO_SIMPLE);
#else
      ncclDevWorkP2p work = {};
      work.sendRank = work.recvRank = -1;
      work.sendBytes = work.recvBytes = count * sizeof(float);
      work.sendAddr = src;
      work.recvAddr = dst;
      work.nP2pChannels = work.nSendChannels = work.nRecvChannels = 1;
      work.sendChunkSize_u32fp8 = work.recvChunkSize_u32fp8 = u32fp8Encode(kChunkBytes);
#if defined(EXAMPLE_SENDRECV)
      work.sendRank = (r + 1) % n;
      work.recvRank = (r + n - 1) % n;
#elif defined(EXAMPLE_ALLTOALL)
      work.sendRank = (r + round) % n;
      work.recvRank = (r + n - round) % n;
      work.sendAddr = src + work.sendRank * count;
      work.recvAddr = dst + work.recvRank * count;
#elif defined(EXAMPLE_GATHER)
      if (r == round) work.sendRank = root;
      if (r == root) {
        work.recvRank = round;
        work.recvAddr = dst + round * count;
      }
#elif defined(EXAMPLE_SCATTER)
      if (r == root) {
        work.sendRank = round;
        work.sendAddr = src + round * count;
      }
      if (r == round) work.recvRank = root;
#endif
#endif
      EXAMPLE_KERNEL<<<1, kThreads, ranks[r].scratchBytes, ranks[r].stream>>>(ranks[r].comm, ranks[r].channel, work);
      CUDA(cudaGetLastError());
    }
  }
  bool passed = true;
  // Complete every rank before inspecting or reinitializing any rank's buffers.
  for (int r = 0; r < n; r++) {
    CUDA(cudaSetDevice(r));
    CUDA(cudaStreamSynchronize(ranks[r].stream));
  }
  for (int r = 0; r < n; r++) {
    CUDA(cudaSetDevice(r));
    Layout x = layout(r, root, count, inPlace);
    float* want = (x.recvOnInput ? inputs[r].data() : outputs[r].data()) + kGuard + x.recvOffset;
    for (size_t i = 0; i < recv_elements(count, n); i++) {
#if defined(EXAMPLE_BROADCAST)
      want[i] = input_value(root, i);
#elif defined(EXAMPLE_REDUCE)
      if (r != root) continue;
      want[i] = 0;
      for (int s = 0; s < n; s++) want[i] += input_value(s, i);
#elif defined(EXAMPLE_ALLGATHER) || defined(EXAMPLE_GATHER)
#if defined(EXAMPLE_GATHER)
      if (r != root) continue;
#endif
      want[i] = input_value(int(i / count), i % count);
#elif defined(EXAMPLE_REDUCESCATTER)
      want[i] = 0;
      for (int s = 0; s < n; s++) want[i] += input_value(s, r * count + i);
#elif defined(EXAMPLE_ALLTOALL)
      want[i] = input_value(int(i / count), r * count + i % count);
#elif defined(EXAMPLE_SCATTER)
      want[i] = input_value(root, r * count + i);
#else
      want[i] = input_value((r + n - 1) % n, i);
#endif
    }
    std::vector<float> actualIn(inSize), actualOut(outSize);
    CUDA(cudaMemcpy(actualIn.data(), ranks[r].input, inSize * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA(cudaMemcpy(actualOut.data(), ranks[r].output, outSize * sizeof(float), cudaMemcpyDeviceToHost));
    passed &= check_buffer("input/guards", r, actualIn, inputs[r]);
    passed &= check_buffer("output/guards", r, actualOut, outputs[r]);
    std::vector<Edge> edges(n);
    std::vector<ncclDevChannelPeer> peers(n);
    CUDA(cudaMemcpy(edges.data(), ranks[r].edges, n * sizeof(Edge), cudaMemcpyDeviceToHost));
    CUDA(cudaMemcpy(peers.data(), ranks[r].peers, n * sizeof(ncclDevChannelPeer), cudaMemcpyDeviceToHost));
    for (int peer = 0; peer < n; peer++) {
      uint64_t recv = (ranks[r].expectedRecv[peer] += edge_delta(peer, r, n, root, count));
      uint64_t send = (ranks[r].expectedSend[peer] += edge_delta(r, peer, n, root, count));
      if (edges[peer].head != recv || edges[peer].tail != recv ||
          peers[peer].recv[kConnIndex].step != recv || peers[peer].send[kConnIndex].step != send) {
        std::fprintf(stderr, "FAIL %s rank=%d peer=%d FIFO/conn.step expected recv=%llu send=%llu\n",
                     kName, r, peer, (unsigned long long)recv, (unsigned long long)send);
        passed = false;
      }
    }
  }
  std::printf("%s %s ranks=%d count=%zu root=%d inplace=%d\n", passed ? "PASS" : "FAIL", kName, n, count, root, inPlace);
  return passed;
}

int main(int argc, char** argv) {
  int n = 2, root = 0;
  size_t count = 4097;
  bool selfTest = false, inPlace = false;
  for (int i = 1; i < argc; i++) {
    if (!std::strcmp(argv[i], "--self-test")) selfTest = true;
    else if (!std::strcmp(argv[i], "--in-place")) inPlace = true;
    else if ((!std::strcmp(argv[i], "--ranks") || !std::strcmp(argv[i], "--count") ||
              !std::strcmp(argv[i], "--root")) && i + 1 < argc) {
      bool isRank = !std::strcmp(argv[i], "--ranks"), isRoot = !std::strcmp(argv[i], "--root");
      const char* arg = argv[++i];
      char* end;
      unsigned long long value = std::strtoull(arg, &end, 10);
      if (arg[0] == '-' || end == arg || *end || value > (isRank || isRoot ? 8ULL : 16777216ULL) || (isRank && !value)) {
        std::fprintf(stderr, "Invalid numeric argument: %s\n", arg);
        return 2;
      }
      if (isRank) n = int(value); else if (isRoot) root = int(value); else count = size_t(value);
    } else {
      std::fprintf(stderr, "Usage: %s [--ranks 1..8] [--count 0..16777216] [--root 0..ranks-1] [--in-place] [--self-test]\n", argv[0]);
      return 2;
    }
  }
  if (root >= n || (!kRooted && root != 0) || (inPlace && !kSupportsInPlace)) {
    std::fprintf(stderr, "Invalid root or unsupported in-place layout for %s\n", kName);
    return 2;
  }
  int available = 0;
  cudaError_t discovery = cudaGetDeviceCount(&available);
  if (discovery != cudaSuccess || available < n) {
    std::fprintf(stderr, "SKIP: requires %d CUDA GPUs (found %d, %s)\n", n, available, cudaGetErrorString(discovery));
    return 77;
  }
  // P2P-composed operations may connect any pair; Ring needs only r -> next.
  for (int r = 0; r < n; r++) {
    CUDA(cudaSetDevice(r));
    for (int p = 0; p < n; p++) {
      if (r == p) continue;
#if !defined(EXAMPLE_P2P) || defined(EXAMPLE_SENDRECV)
      if (p != (r + 1) % n) continue;
#endif
      int accessible;
      CUDA(cudaDeviceCanAccessPeer(&accessible, r, p));
      if (!accessible) {
        std::fprintf(stderr, "SKIP: edge %d -> %d requires CUDA P2P\n", r, p);
        return 77;
      }
    }
  }
  std::vector<size_t> sizes = selfTest ? std::vector<size_t>{0, 1, 17, 1023, 1024, 1025, 4097, 65539, 1048576}
                                     : std::vector<size_t>{count};
  size_t capacity = size_t(n) * *std::max_element(sizes.begin(), sizes.end()) + 2 * kGuard;
  std::vector<Rank> ranks(n);
  for (int r = 0; r < n; r++) {
    CUDA(cudaSetDevice(r));
    for (int p = 0; p < n; p++) {
      if (r == p) continue;
#if !defined(EXAMPLE_P2P) || defined(EXAMPLE_SENDRECV)
      if (p != (r + 1) % n) continue;
#endif
      cudaError_t enabled = cudaDeviceEnablePeerAccess(p, 0);
      if (enabled == cudaErrorPeerAccessAlreadyEnabled) (void)cudaGetLastError(); else CUDA(enabled);
    }
    CUDA(cudaStreamCreateWithFlags(&ranks[r].stream, cudaStreamNonBlocking));
    CUDA(cudaMalloc(&ranks[r].edges, n * sizeof(Edge)));
    CUDA(cudaMemsetAsync(ranks[r].edges, 0, n * sizeof(Edge), ranks[r].stream));
    CUDA(cudaMalloc(&ranks[r].peers, n * sizeof(ncclDevChannelPeer)));
    CUDA(cudaMalloc(&ranks[r].peerPointers, n * sizeof(ncclDevChannelPeer*)));
    CUDA(cudaMalloc(&ranks[r].ringRanks, n * sizeof(int)));
    CUDA(cudaMalloc(&ranks[r].abortFlag, sizeof(uint32_t)));
    CUDA(cudaMemsetAsync(ranks[r].abortFlag, 0, sizeof(uint32_t), ranks[r].stream));
    CUDA(cudaMalloc(&ranks[r].input, capacity * sizeof(float)));
    CUDA(cudaMalloc(&ranks[r].output, capacity * sizeof(float)));
    cudaDeviceProp prop;
    CUDA(cudaGetDeviceProperties(&prop, r));
    ranks[r].scratchBytes = ncclShmemScratchWarpSize(prop.major * 100 + prop.minor * 10) * (kThreads / WARP_SIZE);
    ranks[r].expectedRecv.resize(n);
    ranks[r].expectedSend.resize(n);
  }
  for (int r = 0; r < n; r++) {
    CUDA(cudaSetDevice(r));
    std::vector<ncclDevChannelPeer> peers(n);
    std::vector<ncclDevChannelPeer*> pointers(n);
    std::vector<int> order(n);
    auto connection = [](Edge* edge) {
      ncclConnInfo conn = {};
      conn.buffs[NCCL_PROTO_SIMPLE] = reinterpret_cast<char*>(edge->data);
      conn.head = &edge->head;
      conn.tail = &edge->tail;
      conn.stepSize = kChunkBytes;
      return conn;
    };
    for (int p = 0; p < n; p++) {
      peers[p].recv[kConnIndex] = connection(ranks[r].edges + p);
      peers[p].send[kConnIndex] = connection(ranks[p].edges + r);
      pointers[p] = ranks[r].peers + p;
      order[p] = (r + p) % n;
    }
    CUDA(cudaMemcpy(ranks[r].peers, peers.data(), n * sizeof(ncclDevChannelPeer), cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(ranks[r].peerPointers, pointers.data(), n * sizeof(ncclDevChannelPeer*), cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(ranks[r].ringRanks, order.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    ranks[r].comm.rank = r;
    ranks[r].comm.nRanks = n;
    ranks[r].comm.nNodes = 1;
    ranks[r].comm.buffSizes[NCCL_PROTO_SIMPLE] = NCCL_STEPS * kChunkBytes;
    ranks[r].comm.p2pChunkSize = kChunkBytes;
    ranks[r].comm.abortFlag = ranks[r].abortFlag;
    ranks[r].channel.peers = ranks[r].peerPointers;
    ranks[r].channel.ring.prev = (r + n - 1) % n;
    ranks[r].channel.ring.next = (r + 1) % n;
    ranks[r].channel.ring.index = r;
    ranks[r].channel.ring.userRanks = ranks[r].ringRanks;
  }
  bool passed = true;
  for (size_t size : sizes)
    for (int repeat = 0; repeat < (selfTest ? 2 : 1); repeat++)
      for (int testRoot = 0; testRoot < (selfTest && kRooted ? n : 1); testRoot++) {
        int actualRoot = selfTest && kRooted ? testRoot : root;
        passed &= run_case(ranks, size, selfTest ? false : inPlace, actualRoot);
        if (selfTest && kSupportsInPlace) passed &= run_case(ranks, size, true, actualRoot);
      }
  for (int r = 0; r < n; r++) {
    CUDA(cudaSetDevice(r));
    CUDA(cudaFree(ranks[r].input));
    CUDA(cudaFree(ranks[r].output));
    CUDA(cudaFree(ranks[r].edges));
    CUDA(cudaFree(ranks[r].peers));
    CUDA(cudaFree(ranks[r].peerPointers));
    CUDA(cudaFree(ranks[r].ringRanks));
    CUDA(cudaFree(ranks[r].abortFlag));
    CUDA(cudaStreamDestroy(ranks[r].stream));
  }
  return passed ? 0 : 1;
}

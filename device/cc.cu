// Copyright (c) 2018, A. Barak
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the names of the copyright holders nor the names of its 
//   contributors may be used to endorse or promote products derived from this
//   software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.


#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cinttypes>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <cmath>

#include "benchmark.h"
#include "bitmap.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "pvector.h"
#include "timer.h"
#include <cassert>

#include "cc.h"
#include "cta_scheduler.cuh"

// General helpers
namespace {
  static void HandleError(const char *file, int line, cudaError_t err)
  {
    printf("ERROR in %s:%d: %s (%d)\n", file, line,
      cudaGetErrorString(err), err);
    exit(1);
  }
  // CUDA assertions
#define CUDA_CHECK(err) do {                                  \
    cudaError_t errr = (err);                                 \
    if(errr != cudaSuccess)                                   \
        {                                                     \
        ::HandleError(__FILE__, __LINE__, errr);              \
        }                                                     \
  } while(0)

  static inline __host__ __device__ size_t round_up(
    size_t numerator, size_t denominator)
  {
    return (numerator + denominator - 1) / denominator;
  }

  inline void kernel_sizing(dim3& grid_dims, dim3& block_dims, uint32_t block_size, uint32_t work_size)
  {
    dim3 bd(block_size, 1, 1);
    dim3 gd(round_up(work_size, bd.x), 1, 1);

    grid_dims = gd;
    block_dims = bd;
  }
}

// Device CSRGraph and CSRGraph Allocator 
namespace dev {

  template<typename NodeID_, typename OffsetID_, bool MakeInverse=false>
  struct CSRGraph
  {
    NodeID_ nnodes_;
    OffsetID_ nedges_;
    OffsetID_ *row_start_;
    NodeID_ *edge_dst_;

    CSRGraph(NodeID_ nnodes, OffsetID_ nedges) :
      nnodes_(nnodes), nedges_(nedges), row_start_(nullptr), edge_dst_(nullptr) { }

    CSRGraph() :
      nnodes_(0), nedges_(0), row_start_(nullptr), edge_dst_(nullptr) { }

    __host__ __device__ __forceinline__ NodeID_ num_nodes() const
    {
      return nnodes_;
    }

    __device__ __forceinline__ OffsetID_ begin_edge(NodeID_ node) const
    {
#if __CUDA_ARCH__ >= 320
      return __ldg(row_start_ + node);
#else
      return row_start_[node];
#endif
    }

    __device__ __forceinline__ OffsetID_ end_edge(NodeID_ node) const
    {
#if __CUDA_ARCH__ >= 320
      return __ldg(row_start_ + node + 1);
#else
      return row_start_[node + 1];
#endif
    }

    __device__ __forceinline__ NodeID_ edge_dest(OffsetID_ edge) const
    {
#if __CUDA_ARCH__ >= 320
      return __ldg(edge_dst_ + edge);
#else
      return edge_dst_[edge];
#endif
    }

    // For compatibility with MakeInverse=true
    CSRGraph<NodeID_, OffsetID_, false> GetNoInverseGraph() const { return *this; }

    OffsetID_ ** ptr_to_row_start_in() { return nullptr; }
    NodeID_ **   ptr_to_edge_dst_in() { return nullptr; }

    __device__ __forceinline__ OffsetID_ begin_in_edge(NodeID_ node) const {
      static_assert(MakeInverse, "Inverse not supported");
      return 0;
    }

    __device__ __forceinline__ OffsetID_ end_in_edge(NodeID_ node) const {
      static_assert(MakeInverse, "Inverse not supported"); 
      return 0;
    }

    __device__ __forceinline__ NodeID_ in_edge_src(OffsetID_ edge) const {
      static_assert(MakeInverse, "Inverse not supported"); 
      return 0;
    }
  };

  template<typename NodeID_, typename OffsetID_>
  struct CSRGraph<NodeID_, OffsetID_, true>
  {
    NodeID_ nnodes_;
    OffsetID_ nedges_;
    OffsetID_ *row_start_;
    NodeID_ *edge_dst_;
    OffsetID_ *row_start_in_;
    NodeID_ *edge_dst_in_;

    CSRGraph(NodeID_ nnodes, OffsetID_ nedges) :
      nnodes_(nnodes), nedges_(nedges), row_start_(nullptr), edge_dst_(nullptr), row_start_in_(nullptr), edge_dst_in_(nullptr) { }

    CSRGraph() :
      nnodes_(0), nedges_(0), row_start_(nullptr), edge_dst_(nullptr), row_start_in_(nullptr), edge_dst_in_(nullptr) { }

    __host__ __device__ __forceinline__ NodeID_ num_nodes() const
    {
      return nnodes_;
    }

    __device__ __forceinline__ OffsetID_ begin_edge(NodeID_ node) const
    {
#if __CUDA_ARCH__ >= 320
      return __ldg(row_start_ + node);
#else
      return row_start_[node];
#endif
    }

    __device__ __forceinline__ OffsetID_ end_edge(NodeID_ node) const
    {
#if __CUDA_ARCH__ >= 320
      return __ldg(row_start_ + node + 1);
#else
      return row_start_[node + 1];
#endif
    }

    __device__ __forceinline__ NodeID_ edge_dest(OffsetID_ edge) const
    {
#if __CUDA_ARCH__ >= 320
      return __ldg(edge_dst_ + edge);
#else
      return edge_dst_[edge];
#endif
    }

    //
    // Inverse related methods
    //
    CSRGraph<NodeID_, OffsetID_, false> GetNoInverseGraph() const
    {
      CSRGraph<NodeID_, OffsetID_, false> dev_g(nnodes_, nedges_);
      dev_g.row_start_ = row_start_;
      dev_g.edge_dst_ = edge_dst_;
      return dev_g;
    }

    OffsetID_ ** ptr_to_row_start_in() { return &row_start_in_; }
    NodeID_   ** ptr_to_edge_dst_in() { return &edge_dst_in_; }

    __device__ __forceinline__ OffsetID_ begin_in_edge(NodeID_ node) const
    {
#if __CUDA_ARCH__ >= 320
      return __ldg(row_start_in_ + node);
#else
      return row_start_in_[node];
#endif
    }

    __device__ __forceinline__ OffsetID_ end_in_edge(NodeID_ node) const
    {
#if __CUDA_ARCH__ >= 320
      return __ldg(row_start_in_ + node + 1);
#else
      return row_start_in_[node + 1];
#endif
    }

    __device__ __forceinline__ NodeID_ in_edge_src(OffsetID_ edge) const
    {
#if __CUDA_ARCH__ >= 320
      return __ldg(edge_dst_in_ + edge);
#else
      return edge_dst_in_[edge];
#endif
    }
  };

  template<typename NodeID_, typename OffsetID_, bool MakeInverse=false>
  struct CSRGraphAllocator {
    const ::CSRGraph<NodeID_>& host_g_;
    CSRGraph<NodeID_, OffsetID_, MakeInverse> dev_g_;
    CSRGraphAllocator(const ::CSRGraph<NodeID_>& host_g) : host_g_(host_g) {
      if (host_g_.num_edges_directed() > (std::numeric_limits<OffsetID_>::max)()) {
        printf("Warning: Number of edges too large for OffsetID_ type used\n");
        std::exit(-31);
      }
      Allocate();
      cudaDeviceSynchronize();
    }

    ~CSRGraphAllocator() {
      Deallocate();
    }

    void Allocate() {
      NodeID_ nnodes;
      OffsetID_ nedges;

      dev_g_.nnodes_ = nnodes = host_g_.num_nodes();
      dev_g_.nedges_ = nedges = host_g_.num_edges_directed();

      // Extract offsets from pointers
      pvector<OffsetID_> out_offsets(host_g_.num_nodes() + 1);
      const NodeID* g_out_start = host_g_.out_neigh(0).begin();
      out_offsets[0] = 0;
#pragma omp parallel for 
      for (NodeID_ n = 1; n < host_g_.num_nodes() + 1; n++)
      {
        out_offsets[n] = host_g_.out_neigh(n - 1).end() - g_out_start;
      }

      CUDA_CHECK(cudaMalloc((void**)&dev_g_.row_start_, (nnodes + 1) * sizeof(OffsetID_))); // Malloc and copy +1 for the row_start's extra cell
      CUDA_CHECK(cudaMemcpy(dev_g_.row_start_, out_offsets.data(), (nnodes + 1) * sizeof(OffsetID_), cudaMemcpyHostToDevice));

      CUDA_CHECK(cudaMalloc((void**)&dev_g_.edge_dst_, nedges * sizeof(NodeID_)));
      CUDA_CHECK(cudaMemcpy(dev_g_.edge_dst_, g_out_start, nedges * sizeof(NodeID_), cudaMemcpyHostToDevice));

      if (MakeInverse)
      {
        // Extract offsets from pointers
        pvector<OffsetID_> in_offsets(host_g_.num_nodes() + 1);
        const NodeID* g_in_start = host_g_.in_neigh(0).begin();
        in_offsets[0] = 0;
#pragma omp parallel for 
        for (NodeID_ n = 1; n < host_g_.num_nodes() + 1; n++)
        {
          in_offsets[n] = host_g_.in_neigh(n - 1).end() - g_in_start;
        }

        CUDA_CHECK(cudaMalloc((void**)dev_g_.ptr_to_row_start_in(), (nnodes + 1) * sizeof(OffsetID_))); // Malloc and copy +1 for the row_start's extra cell
        CUDA_CHECK(cudaMemcpy(*dev_g_.ptr_to_row_start_in(), in_offsets.data(), (nnodes + 1) * sizeof(OffsetID_), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc((void**)dev_g_.ptr_to_edge_dst_in(), nedges * sizeof(NodeID_)));
        CUDA_CHECK(cudaMemcpy(*dev_g_.ptr_to_edge_dst_in(), g_in_start, nedges * sizeof(NodeID_), cudaMemcpyHostToDevice));
      }

      //printf("%p, %p\n", dev_g_.row_start_, dev_g_.edge_dst_);
    }

    void Deallocate()
    {
      //printf("%p, %p\n", dev_g_.row_start_, dev_g_.edge_dst_);

      CUDA_CHECK(cudaFree(dev_g_.row_start_));
      CUDA_CHECK(cudaFree(dev_g_.edge_dst_));

      dev_g_.row_start_ = nullptr;
      dev_g_.edge_dst_ = nullptr;

      if (MakeInverse)
      {
        CUDA_CHECK(cudaFree(*dev_g_.ptr_to_row_start_in()));
        CUDA_CHECK(cudaFree(*dev_g_.ptr_to_edge_dst_in()));

        *dev_g_.ptr_to_row_start_in() = nullptr;
        *dev_g_.ptr_to_edge_dst_in() = nullptr;
      }
    }
  };
}

typedef int64_t OffsetID;

typedef dev::CSRGraph<NodeID, OffsetID> DevGraph;
typedef dev::CSRGraphAllocator<NodeID, OffsetID> DevGraphAllocator;

typedef dev::CSRGraph<NodeID, OffsetID, true> DevGraphInverse;
typedef dev::CSRGraphAllocator<NodeID, OffsetID, true> DevGraphInverseAllocator;

#define BLOCK_THREADS 256
#define SAMPLE_SIZE 1024

#define TID_1D (threadIdx.x + blockIdx.x * blockDim.x)
#define TID_1D_64 (threadIdx.x + (int64_t)blockIdx.x * blockDim.x)
#define TOTAL_THREADS_1D (gridDim.x * blockDim.x)
#define TOTAL_THREADS_1D_64 ((int64_t)gridDim.x * blockDim.x)

// Afforest GPU kernels
namespace kernels {

  __global__ void InitKernel(NodeID* comp, NodeID nnodes)
  {
    NodeID n = TID_1D;
    if (n >= nnodes) return;
    comp[n] = n;
  }

  __device__ __forceinline__ void Link(NodeID* comp, NodeID u, NodeID v)
  {
    NodeID p1 = comp[u];
    NodeID p2 = comp[v];

    while (p1 != p2)
    {
      NodeID high = p1 > p2 ? p1 : p2;
      NodeID low = p1 + (p2 - high);

      // atomicCAS(ptr, compare, val);
      // TODO: test precheck
      int prev = atomicCAS(comp + high, high, low);

      if (prev == high || prev == low) {
        break;
      }

      p1 = comp[comp[high]];
      p2 = comp[low];
    }
  }

  __global__ void LinkKernel(DevGraph g, NodeID* comp)
  {
    uint32_t tid = TID_1D;
    uint32_t nthreads = TOTAL_THREADS_1D;

    NodeID roundedup = round_up(g.nnodes_, blockDim.x) * blockDim.x; // We want all threads in active blocks to enter the loop

    for (NodeID n = tid; n < roundedup; n += nthreads)
    {
      CTA::np_local<NodeID> np_local = { 0, 0, 0 };

      if (n < g.nnodes_)
      {
        np_local.start = g.begin_edge(n);
        np_local.size = g.end_edge(n) - np_local.start;
        np_local.meta_data = n;
      }

      CTA::CTAWorkScheduler<NodeID>::template schedule(
        np_local,
        [&g, &comp](OffsetID edge, NodeID u)
        {
          NodeID v = g.edge_dest(edge);
          Link(comp, u, v);
        }
      );
    }
  }

  __global__ void LinkNeighborKernel(DevGraph g, NodeID* comp, NodeID neighbor)
  {
    NodeID u = TID_1D;
    if (u >= g.nnodes_) return;

    OffsetID edge = g.begin_edge(u) + neighbor;
    if (edge < g.end_edge(u))
    {
      NodeID v = g.edge_dest(edge);
      Link(comp, u, v);
    }
  }

  __global__ void SampleKernel(NodeID* comp, NodeID* samples)
  {
    uint32_t tid = TID_1D;
    samples[tid] = comp[samples[tid]]; // Note: samples are updated in place
  }

  __global__ void LinkSkipKernel(DevGraph g, NodeID* comp, NodeID start_neighbor, NodeID skip)
  {
    uint32_t tid = TID_1D;
    uint32_t nthreads = TOTAL_THREADS_1D;

    NodeID roundedup = round_up(g.nnodes_, blockDim.x) * blockDim.x; // we want all threads in active blocks to enter the loop

    for (NodeID n = tid; n < roundedup; n += nthreads)
    {
      CTA::np_local<NodeID> np_local = { 0, 0, 0 };

      if (n < g.nnodes_ && comp[n] != skip)
      {
        OffsetID begin = g.begin_edge(n) + start_neighbor, end = g.end_edge(n);
        np_local.start = begin;
        np_local.size = begin < end ? end - begin : 0;
        np_local.meta_data = n;
      }

      CTA::CTAWorkScheduler<NodeID>::template schedule(
        np_local,
        [&g, &comp](OffsetID edge, NodeID u)
        {
          NodeID v = g.edge_dest(edge);
          Link(comp, u, v);
        }
      );
    }
  }

  //
  // Should be used only for directed graphs and during the skip link phase
  //
  __global__ void LinkSkipInverseKernel(DevGraphInverse g, NodeID* comp, NodeID start_neighbor, NodeID skip)
  {
    uint32_t tid = TID_1D;
    uint32_t nthreads = TOTAL_THREADS_1D;

    NodeID roundedup = round_up(g.nnodes_, blockDim.x) * blockDim.x; // We want all threads in active blocks to enter the loop

    for (NodeID n = tid; n < roundedup; n += nthreads)
    {
      CTA::np_local<NodeID> np_local = { 0, 0, 0 };

      bool has_work = n < g.nnodes_ && comp[n] != skip;

      if (has_work)
      {
        OffsetID begin = g.begin_edge(n) + start_neighbor, end = g.end_edge(n);
        np_local.start = begin;
        np_local.size = begin < end ? end - begin : 0;
        np_local.meta_data = n;
      }

      CTA::CTAWorkScheduler<NodeID>::template schedule(
        np_local,
        [&g, &comp](OffsetID edge, NodeID u)
        {
          NodeID v = g.edge_dest(edge);
          Link(comp, u, v);
        }
      );

      // Go over incoming edges as well
      if (has_work)
      {
        OffsetID begin = g.begin_in_edge(n), end = g.end_in_edge(n);
        np_local.start = begin;
        np_local.size = begin < end ? end - begin : 0;
        np_local.meta_data = n;
      }

      CTA::CTAWorkScheduler<NodeID>::template schedule(
        np_local,
        [&g, &comp](OffsetID edge, NodeID u)
        {
          NodeID v = g.in_edge_src(edge);
          Link(comp, u, v);
        }
      );
    }
  }

  __global__ void CompressKernel(NodeID* comp, NodeID nnodes)
  {
    NodeID n = TID_1D;
    if (n >= nnodes) return;

    NodeID p, pp;

    p = comp[n];
    pp = comp[p];

    while (p != pp)
    {
      comp[n] = pp;
      p = pp;
      pp = comp[p];
    }
  }
}

using namespace kernels;

// Afforest host-side runners
namespace {
  std::pair<NodeID, float> SampleLargestStar(NodeID* dev_comp, NodeID* dev_samples, NodeID nnodes)
  {
    pvector<NodeID> samples(SAMPLE_SIZE);

    // Select random nodes
    std::mt19937 gen;
    std::uniform_int_distribution<NodeID> distribution(0, nnodes - 1);
    for (NodeID i = 0; i < SAMPLE_SIZE; i++) {
      samples[i] = distribution(gen);
    }

    cudaStream_t copy_stream;
    cudaEvent_t copy_event;
    CUDA_CHECK(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&copy_event, cudaEventDisableTiming));

    CUDA_CHECK(cudaMemcpyAsync(dev_samples, samples.data(), SAMPLE_SIZE * sizeof(NodeID), cudaMemcpyHostToDevice, copy_stream));
    cudaEventRecord(copy_event, copy_stream);
    cudaStreamWaitEvent(nullptr, copy_event, 0);

    dim3 grid_dims, block_dims;
    kernel_sizing(grid_dims, block_dims, BLOCK_THREADS, SAMPLE_SIZE);
    SampleKernel << < grid_dims, block_dims >> > (dev_comp, dev_samples);

    CUDA_CHECK(cudaMemcpyAsync(samples.data(), dev_samples, SAMPLE_SIZE * sizeof(NodeID), cudaMemcpyDeviceToHost, nullptr));
    cudaStreamSynchronize(nullptr);

    std::unordered_map<NodeID, NodeID> sample_counts(SAMPLE_SIZE);
    using pair_type = std::unordered_map<NodeID, NodeID>::value_type;

    for (NodeID i = 0; i < SAMPLE_SIZE; i++)
    {
      auto it = sample_counts.find(samples[i]);
      if (it == sample_counts.end())  sample_counts[samples[i]] = 1;
      else                            ++it->second;
    }

    auto sample_maxstar = std::max_element(
      sample_counts.begin(), sample_counts.end(),
      [](const pair_type& p1, const pair_type& p2) { return p1.second < p2.second; });

    CUDA_CHECK(cudaStreamDestroy(copy_stream));
    CUDA_CHECK(cudaEventDestroy(copy_event));

    return std::make_pair(sample_maxstar->first, static_cast<float>(sample_maxstar->second) / SAMPLE_SIZE);
  }

  void RunAfforestSkip(const Graph& host_g, DevGraph& dev_g, NodeID* dev_comp, NodeID* dev_samples, NodeID neighbor_rounds = 2)
  {
    NodeID nnodes = dev_g.nnodes_;
    dim3 grid_dims, block_dims;
    kernel_sizing(grid_dims, block_dims, BLOCK_THREADS, nnodes);

    InitKernel << < grid_dims, block_dims >> > (dev_comp, nnodes);

    for (NodeID neighbor = 0; neighbor < neighbor_rounds; neighbor++)
    {
      LinkNeighborKernel << < grid_dims, block_dims >> > (dev_g, dev_comp, neighbor);
      CompressKernel << < grid_dims, block_dims >> > (dev_comp, nnodes);
    }

    auto largest_star = SampleLargestStar(dev_comp, dev_samples, nnodes);
    NodeID skip = largest_star.first;
    printf("Skipping largest sampled component %d (%1.2f %% of undirected graph)\n", skip, largest_star.second*100);

    LinkSkipKernel << < grid_dims, block_dims >> > (dev_g, dev_comp, neighbor_rounds, skip);
    CompressKernel << < grid_dims, block_dims >> > (dev_comp, nnodes);
  }

  void RunAfforestSkipInverse(const Graph& host_g, DevGraphInverse& dev_g, NodeID* dev_comp, NodeID* dev_samples, NodeID neighbor_rounds = 2)
  {
    NodeID nnodes = dev_g.nnodes_;
    dim3 grid_dims, block_dims;
    kernel_sizing(grid_dims, block_dims, BLOCK_THREADS, nnodes);

    InitKernel << < grid_dims, block_dims >> > (dev_comp, nnodes);

    for (NodeID neighbor = 0; neighbor < neighbor_rounds; neighbor++)
    {
      LinkNeighborKernel << < grid_dims, block_dims >> > (dev_g.GetNoInverseGraph(), dev_comp, neighbor);
      CompressKernel << < grid_dims, block_dims >> > (dev_comp, nnodes);
    }

    NodeID skip = SampleLargestStar(dev_comp, dev_samples, nnodes).first;

    LinkSkipInverseKernel << < grid_dims, block_dims >> > (dev_g, dev_comp, neighbor_rounds, skip);
    CompressKernel << < grid_dims, block_dims >> > (dev_comp, nnodes);
  }
}

void BenchmarkConnectedComponents(CLApp& cli, const Graph& g)
{
  NodeID nnodes = g.num_nodes();
  pvector<NodeID> host_comp(nnodes);

  NodeID* dev_comp;
  CUDA_CHECK(cudaMalloc((void**)&dev_comp, nnodes*sizeof(NodeID)));

  NodeID* dev_samples; // Used only if skip is on
  CUDA_CHECK(cudaMalloc((void**)&dev_samples, SAMPLE_SIZE*sizeof(NodeID)));

  // Shared memory usage is minor 
  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
  cudaDeviceSynchronize();

  if (g.directed()) {
    // Allocating inverse graph for supporting skip on directed graphs
    DevGraphInverseAllocator dev_allocator(g);
    DevGraphInverse& dev_g = dev_allocator.dev_g_;
    BenchmarkKernel(cli, [&]() { RunAfforestSkipInverse(g, dev_g, dev_comp, dev_samples); cudaDeviceSynchronize(); });
  }
  else {
    DevGraphAllocator dev_allocator(g);
    DevGraph& dev_g = dev_allocator.dev_g_;
    BenchmarkKernel(cli, [&](){ RunAfforestSkip(g, dev_g, dev_comp, dev_samples); cudaDeviceSynchronize(); });
  }

  CUDA_CHECK(cudaMemcpy(host_comp.data(), dev_comp, nnodes*sizeof(NodeID), cudaMemcpyDeviceToHost));
  
  if (cli.do_analysis()) {
    PrintCompStats(g, host_comp);
  }
  if (cli.do_verify()) {
    PrintLabel("Verification", CCVerifier(g, host_comp) ? "PASS" : "FAIL");
  }

  CUDA_CHECK(cudaFree(dev_comp));
  CUDA_CHECK(cudaFree(dev_samples));
}

int main(int argc, char* argv[]) {
  CLApp cli(argc, argv, "GPU-connected-components");
  if (!cli.ParseArgs())
    return -1;
  Builder b(cli);
  Graph g = b.MakeGraph();
  g.PrintStats();

  BenchmarkConnectedComponents(cli, g);

  // cudaDeviceReset must be called before exiting in order for profiling and
  // tracing tools such as Nsight and Visual Profiler to show complete traces.
  cudaError_t cudaStatus = cudaDeviceReset();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaDeviceReset failed!");
    return 1;
  }

  return 0;
}
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

#include <algorithm>
#include <cinttypes>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "benchmark.h"
#include "bitmap.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "pvector.h"
#include "timer.h"
#include "cc.h"


/*
GAP Benchmark Suite
Kernel: Connected Components (CC)
Author: Michael Sutton

Will return comp array labelling each vertex with a connected component ID

This CC implementation makes use of the Afforest subgraph sampling algorithm [1],
which restructures and extends the Shiloach-Vishkin [2] algorithm.

[1] Michael Sutton, Tal Ben-Nun, and Amnon Barak. "Optimizing Parallel 
    Graph Connectivity Computation via Subgraph Sampling" Symposium on 
    Parallel and Distributed Processing, IPDPS 2018.

[2] Yossi Shiloach and Uzi Vishkin. "An o(logn) parallel connectivity algorithm"
    Journal of Algorithms, 3(1):57–67, 1982.
*/


__FORCEINLINE__
void Link(NodeID u, NodeID v, pvector<NodeID>& comp) {
  NodeID p1 = comp[u];
  NodeID p2 = comp[v];
  while (p1 != p2) {
    NodeID high = p1 > p2 ? p1 : p2;
    NodeID low = p1 + (p2 - high);
    NodeID p_high = comp[high];
    if ((p_high == low) ||                                              // Was already 'low'
        (p_high == high && compare_and_swap(comp[high], high, low)))    // Succeeded on writing 'low'  
      break;
    p1 = comp[comp[high]];
    p2 = comp[low];
  }
}


void Compress(const Graph &g, pvector<NodeID>& comp) {
  #pragma omp parallel for schedule(static, 2048)
  for (NodeID n = 0; n < g.num_nodes(); n++) {
    while (comp[n] != comp[comp[n]]) {
      comp[n] = comp[comp[n]];
    }
  }
}


NodeID SampleFrequentElement(const pvector<NodeID>& comp, int64_t num_samples = 1024) {
  std::unordered_map<NodeID, int> sample_counts(32);
  using kvp_type = std::unordered_map<NodeID, int>::value_type;
  // Sample elements from 'comp'
  std::mt19937 gen;
  std::uniform_int_distribution<NodeID> distribution(0, comp.size() - 1);
  for (NodeID i = 0; i < num_samples; i++) {
    NodeID n = distribution(gen);
    auto it = sample_counts.find(comp[n]);
    if (it == sample_counts.end())  
      sample_counts[comp[n]] = 1;
    else
      ++it->second;
  }
  // Find an estimation for the most frequent element 
  auto most_frequent = std::max_element(
    sample_counts.begin(), sample_counts.end(),
    [](const kvp_type& p1, const kvp_type& p2) { return p1.second < p2.second; });
  std::cout
    << "Skipping largest intermediate component (ID: " << most_frequent->first
    << ", approx. " << static_cast<int>((static_cast<float>(most_frequent->second) / num_samples) * 100)
    << "% of the graph)" << std::endl;
  return most_frequent->first;
}


pvector<NodeID> Afforest(const Graph &g, int32_t neighbor_rounds = 2)
{
  pvector<NodeID> comp(g.num_nodes());

  // Initialize each node to a single-node self-pointing tree
  #pragma omp parallel for
  for (NodeID n = 0; n < g.num_nodes(); n++)
    comp[n] = n;

  // Process a sparse sampled subgraph first for approximating components.
  // The sampling is done by processing a fixed number of neighbors for each node (see paper)
  for (int r = 0; r < neighbor_rounds; ++r) {
    #pragma omp parallel for
    for (NodeID u = 0; u < g.num_nodes(); u++) {
      NodeID v;
      if (g.out_neigh(u, r, v)) {
        Link(u, v, comp);
      }
    }
    Compress(g, comp);
  }

  // Sample 'comp' to find the most frequent element -- due to prior compression, this 
  // value represents the largest intermediate component
  NodeID c = SampleFrequentElement(comp);

  // Perform a final 'link' phase over remaining edges (excluding largest component) 
  if (g.directed() == false) {
    #pragma omp parallel for schedule(dynamic, 2048)
    for (NodeID u = 0; u < g.num_nodes(); u++) {
      if (comp[u] == c) continue; // Skip processing nodes from the largest component
      for (NodeID v : g.out_neigh(u, neighbor_rounds)) { // Start from the correct neighbor
        Link(u, v, comp);
      }
    }
  }
  else {
    // The algorithm supports finding the Weakly Connected Components (WCC) for directed
    // graphs as well. However, in order to support skipping of large component, incoming  
    // edges of other components must be processed as well
    #pragma omp parallel for schedule(dynamic, 2048)
    for (NodeID u = 0; u < g.num_nodes(); u++) {
      if (comp[u] == c) continue; 
      for (NodeID v : g.out_neigh(u, neighbor_rounds)) { 
        Link(u, v, comp);
      }
      for (NodeID v : g.in_neigh(u)) { // Process parts of the reverse graph as well
        Link(u, v, comp);
      }
    }
  }
  // Finally, 'compress' for final convergence 
  Compress(g, comp);  
  return comp;
}


int main(int argc, char* argv[]) {
  CLApp cli(argc, argv, "connected-components-afforest");
  if (!cli.ParseArgs())
    return -1;
  Builder b(cli);
  Graph g = b.MakeGraph();
  BenchmarkKernel(cli, g, [&](const Graph& gr){ return Afforest(gr); }, PrintCompStats, CCVerifier);
  return 0;
}

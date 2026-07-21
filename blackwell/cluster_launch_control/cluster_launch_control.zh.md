# 4.12. 使用 Cluster Launch Control 实现工作窃取（Work Stealing）

> 来源：<https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cluster-launch-control.html>

在开发 CUDA 应用时，处理数据量和计算量可变的问题至关重要。传统上，CUDA 开发者用两种主要方式来决定启动多少 kernel threadblock：*每 threadblock 固定工作量（fixed work per thread block）* 和 *固定 threadblock 数量（fixed number of thread blocks）*。两种方式各有优劣。

**每 threadblock 固定工作量：** 这种方式中，threadblock 的数量由问题规模决定，而每个 threadblock 所做的工作量保持恒定。

该方式的主要优势：

* *SM 之间的负载均衡*

  当 threadblock 的运行时间存在波动、和/或 threadblock 的数量远大于 GPU 能同时执行的数量（产生 low-tail 效应）时，这种方式允许 GPU 调度器在某些 SM 上比其他 SM 运行更多的 threadblock。
* *抢占（Preemption）*

  即使一个[高优先级 kernel](../02-basics/asynchronous-execution.html#async-execution-stream-priorities) 在某个低优先级 kernel 已经开始执行之后才启动，GPU 调度器也能在低优先级 kernel 的 threadblock 陆续完成时，把高优先级 kernel 的 threadblock 调度上去，从而开始执行它。待高优先级 kernel 执行完毕后，再恢复执行低优先级 kernel。

**固定 threadblock 数量：** 这种方式通常实现为 block-stride 或 grid-stride 循环，threadblock 的数量不依赖于问题规模；相反，每个 threadblock 所做的工作量是问题规模的函数。通常，threadblock 的数量基于运行该 kernel 的 GPU 上的 SM 数量以及期望的 occupancy 来确定。

该方式的主要优势：

* *降低 threadblock 开销*

  这种方式不仅降低了摊薄后的 threadblock 启动延迟，还最小化了与所有 threadblock 共有操作相关的计算开销。这些开销可能显著高于启动延迟开销。

  例如，在卷积 kernel 中，用于计算卷积系数的前导（prologue）——它与 threadblock 索引无关——由于 threadblock 数量固定，可以少计算几次，从而减少冗余计算。

**Cluster Launch Control（CLC）** 是 NVIDIA Blackwell GPU 架构（计算能力 10.0）引入的特性，旨在结合上述两种方式的优点。它通过允许开发者取消（cancel）threadblock 或 threadblock cluster，赋予开发者对 threadblock 调度更多的控制。该机制实现了**工作窃取（work stealing）**。工作窃取是并行计算中的一种动态负载均衡技术：空闲的处理器主动从繁忙处理器的工作队列中「窃取」任务，而不是等待任务被分配。

![Cluster Launch Control 执行流程](images/cluster_launch_control.png)

*图 54：Cluster Launch Control 执行流程*

借助 cluster launch control，一个 threadblock 尝试取消另一个尚未开始执行的 threadblock 的启动。如果取消请求成功，它就通过使用被取消 threadblock 的索引来执行任务，从而「窃取」了对方的工作。如果没有更多可用的 threadblock 索引，或出于其他原因（例如有高优先级 kernel 被调度），取消会失败。在后一种情况下，如果一个 threadblock 在取消失败后退出，调度器就能开始执行高优先级 kernel，之后再继续调度当前 kernel 剩余的 threadblock 执行。上图展示了这一过程的执行流程。

下表总结了三种方式的优缺点：

|  | **每 threadblock 固定工作量** | **固定 threadblock 数量** | **Cluster Launch Control** |
| --- | --- | --- | --- |
| 降低开销 | ❌ | ✅ | ✅ |
| 抢占 | ✅ | ❌ | ✅ |
| 负载均衡 | ✅ | ❌ | ✅ |

## 4.12.1. API 细节

通过 cluster launch control API 取消一个 threadblock 是**异步**完成的，并使用共享内存 barrier 来同步，其编程模式类似于[异步数据拷贝](../03-advanced/advanced-kernel-programming.html#advanced-kernels-async-copies)。

该 API 通过 [libcu++](https://nvidia.github.io/cccl/unstable/libcudacxx/ptx_api.html) 提供：

* 一条**请求指令**，它把编码后的取消结果写入一个 `__shared__` 变量。
* 若干**解码指令**，用于提取成功/失败状态以及被取消的 threadblock 索引。

注意，cluster launch control 操作被建模为 async proxy 操作（参见 [Async Thread and Async Proxy](../03-advanced/advanced-kernel-programming.html#advanced-kernels-hardware-implementation-asynchronous-execution-features-async-thread-proxy)）。

### 4.12.1.1. Threadblock 取消（Cancellation）

使用 Cluster Launch Control 的推荐方式是从单个线程发起，即一次一个请求。

取消过程包含五个步骤：

* **设置阶段**（步骤 1-2）：声明并初始化取消结果和同步变量。
* **工作窃取循环**（步骤 3-5）：反复执行以请求、同步和处理取消结果。

1. 声明用于 threadblock 取消的变量：

   ```cpp
   __shared__ uint4 result; // 请求结果。
   __shared__ uint64_t bar; // 同步 barrier。
   int phase = 0;           // 同步 barrier 的 phase。
   ```
2. 用单个 arrival count 初始化共享内存 barrier：

   ```cpp
   if (cg::thread_block::thread_rank() == 0)
       ptx::mbarrier_init(&bar, 1);
   __syncthreads();
   ```
3. 由单个线程提交异步取消请求，并设置 transaction count：

   ```cpp
   if (cg::thread_block::thread_rank() == 0) {
       cg::invoke_one(cg::coalesced_threads(), [&](){ptx::clusterlaunchcontrol_try_cancel(&result, &bar);});
       ptx::mbarrier_arrive_expect_tx(ptx::sem_relaxed, ptx::scope_cta, ptx::space_shared, &bar, sizeof(uint4));
   }
   ```

   > **注意：** 由于 threadblock 取消是一条 uniform 指令，建议在 [`invoke_one`](cooperative-groups.html#cooperative-groups-invoke-one) 线程选择器内部提交它。这样编译器就能优化掉 peeling 循环。
4. 同步（完成）异步取消请求：

   ```cpp
   while (!ptx::mbarrier_try_wait_parity(&bar, phase))
   {}
   phase ^= 1;
   ```
5. 获取取消状态和被取消的 threadblock 索引：

   ```cpp
   bool success = ptx::clusterlaunchcontrol_query_cancel_is_canceled(result);
   if (success) {
       // 对 1D/2D threadblock 而言不需要全部三个：
       int bx = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_x(result);
       int by = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_y(result);
       int bz = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_z(result);
   }
   ```
6. 确保共享内存操作在 async proxy 与 generic [proxy](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#proxies) 之间的可见性，并防止工作窃取循环各次迭代之间的数据竞争。

### 4.12.1.2. Threadblock 取消的约束

这些约束都与**失败的**取消请求有关：

* 在**观察到（observing）**一个先前失败的请求之后，再提交另一个取消请求，是*未定义行为（undefined behavior）*。

  在下面两个代码示例中，假设第一个取消请求失败，只有第一个示例表现出未定义行为。第二个示例是正确的，因为两个取消请求之间没有「观察」动作：

  **无效代码：**

  ```cpp
  // 第一个请求：
  ptx::clusterlaunchcontrol_try_cancel(&result0, &bar0);

  // 第一个请求的查询：
  [此处为同步 bar0 的代码。]
  bool success0 = ptx::clusterlaunchcontrol_query_cancel_is_canceled(result0);
  assert(!success0); // 观察到失败；后续的第二次取消将无效。

  // 第二个请求 —— 下一行是未定义行为：
  ptx::clusterlaunchcontrol_try_cancel(&result1, &bar1);
  ```

  **有效代码：**

  ```cpp
  // 第一个请求：
  ptx::clusterlaunchcontrol_try_cancel(&result0, &bar0);

  // 第二个请求：
  ptx::clusterlaunchcontrol_try_cancel(&result1, &bar1);

  // 第一个请求的查询：
  [此处为同步 bar0 的代码。]
  bool success0 = ptx::clusterlaunchcontrol_query_cancel_is_canceled(result0);
  assert(!success0); // 观察到失败；但第二次取消是有效的。
  ```
* 获取一个**失败的**取消请求的 threadblock 索引，是未定义行为。
* 不推荐从多个线程提交取消请求。这会导致取消多个 threadblock，并需要谨慎处理，例如：

  + 每个提交的线程必须提供唯一的 `__shared__` result 指针，以避免数据竞争。
  + 如果用同一个 barrier 来同步，则必须相应地调整 arrival count 和 transaction count。

## 4.12.2. 示例：向量-标量乘法

在以下小节中，我们用一个向量-标量乘法 kernel 来演示通过 cluster launch control 实现的工作窃取。我们展示同一问题的两个变体：一个使用 threadblock，一个使用 threadblock cluster。

### 4.12.2.1. 用例：Threadblock

下面三个 kernel 分别演示向量-标量乘法 `v := α·v` 的 *每 threadblock 固定工作量*、*固定 threadblock 数量* 和 *Cluster Launch Control* 三种方式。

* 每 threadblock 固定工作量：

  ```cpp
  __global__
  void kernel_fixed_work (float* data, int n)
  {
      // 前导：
      float alpha = compute_scalar();

      // 计算：
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < n)
          data[i] *= alpha;
  }

  // 启动: kernel_fixed_work<<<(n + 1023) / 1024, 1024>>>(data, n);
  ```
* 固定 threadblock 数量：

  ```cpp
  __global__
  void kernel_fixed_blocks (float* data, int n)
  {
      // 前导：
      float alpha = compute_scalar();

      // 计算：
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      while (i < n) {
          data[i] *= alpha;
          i += gridDim.x * blockDim.x;
      }
  }

  // 启动: kernel_fixed_blocks<<<SM_COUNT, 1024>>>(data, n);
  ```
* Cluster Launch Control：

  ```cpp
  #include <cooperative_groups.h>
  #include <cuda/ptx>

  namespace cg = cooperative_groups;
  namespace ptx = cuda::ptx;

  __global__
  void kernel_cluster_launch_control (float* data, int n)
  {
      // cluster launch control 初始化：
      __shared__ uint4 result;
      __shared__ uint64_t bar;
      int phase = 0;

      if (cg::thread_block::thread_rank() == 0)
          ptx::mbarrier_init(&bar, 1);

      // 前导：
      float alpha = compute_scalar(); // 此代码片段未展示该设备函数。

      // 工作窃取循环：
      int bx = blockIdx.x; // 假设为 1D x 轴 threadblock。

      while (true) {
          // 防止 result 在下一次迭代中被覆盖，
          // （同时确保第 1 次迭代时 barrier 已初始化）：
          __syncthreads();

          // 取消请求：
          if (cg::thread_block::thread_rank() == 0) {
              // 在 async proxy 中 acquire 对 result 的写：
              ptx::fence_proxy_async_generic_sync_restrict(ptx::sem_acquire, ptx::space_cluster, ptx::scope_cluster);

              cg::invoke_one(cg::coalesced_threads(), [&](){ptx::clusterlaunchcontrol_try_cancel(&result, &bar);});
              ptx::mbarrier_arrive_expect_tx(ptx::sem_relaxed, ptx::scope_cta, ptx::space_shared, &bar, sizeof(uint4));
          }

          // 计算：
          int i = bx * blockDim.x + threadIdx.x;
          if (i < n)
              data[i] *= alpha;

          // 取消请求同步：
          while (!ptx::mbarrier_try_wait_parity(ptx::sem_acquire, ptx::scope_cta, &bar, phase))
          {}
          phase ^= 1;

          // 取消请求解码：
          bool success = ptx::clusterlaunchcontrol_query_cancel_is_canceled(result);
          if (!success)
              break;

          bx = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_x<int>(result);

          // 向 async proxy release 对 result 的读：
          ptx::fence_proxy_async_generic_sync_restrict(ptx::sem_release, ptx::space_shared, ptx::scope_cluster);
      }
  }

  // 启动: kernel_cluster_launch_control<<<(n + 1023) / 1024, 1024>>>(data, n);
  ```

### 4.12.2.2. 用例：Threadblock Cluster

对于 [threadblock cluster](../02-basics/intro-to-cuda-cpp.html#thread-block-clusters)，threadblock 取消的步骤与非 cluster 场景相同，只需做少量调整。与非 cluster 情况一样，不推荐从**一个 cluster 内**的多个线程提交取消请求，因为这会尝试取消多个 cluster。

* 取消由单个 cluster 线程提交。
* 每个 cluster 的 threadblock 的共享内存 result 都会收到相同的（编码后的）被取消 threadblock 索引值（即 result 值被 multicast 广播）。所有 threadblock 收到的 result 对应于 cluster 内的局部块索引 `{0, 0, 0}`。因此，cluster 内的各 threadblock 需要加上自己的局部块索引。
* 同步由每个 cluster 的 threadblock 使用一个局部 `__shared__` 内存 barrier 来完成。barrier 操作必须以 `ptx::scope_cluster` 作用域执行。
* 在 cluster 情况下取消要求所有 threadblock 都已存在。用户可以通过 [sync](../05-appendices/device-callable-apis.html#cg-api-sync-function) API 的 `cg::cluster_group::sync()` 来保证所有 threadblock 都在运行。

下面的 kernel 演示了使用 threadblock cluster 的 cluster launch control 方式。

```cpp
#include <cooperative_groups.h>
#include <cuda/ptx>

namespace cg = cooperative_groups;
namespace ptx = cuda::ptx;

__global__ __cluster_dims__(2, 1, 1)
void kernel_cluster_launch_control (float* data, int n)
{
    // cluster launch control 初始化：
    __shared__ uint4 result;
    __shared__ uint64_t bar;
    int phase = 0;

    if (cg::thread_block::thread_rank() == 0) {
        ptx::mbarrier_init(&bar, 1);
        ptx::fence_mbarrier_init(ptx::sem_release, ptx::scope_cluster); // CGA 级 fence。
    }

    // 前导：
    float alpha = compute_scalar(); // 此代码片段未展示该设备函数。

    // 工作窃取循环：
    int bx = blockIdx.x; // 假设为 1D x 轴 threadblock。

    while (true) {
        // 防止 result 在下一次迭代中被覆盖，
        // （同时确保第 1 次迭代时所有 threadblock 都已启动）：
        cg::cluster_group::sync();

        // 由单个 cluster 线程提交取消请求：
        if (cg::cluster_group::thread_rank() == 0) {
            // 在 async proxy 中 acquire 对 result 的写：
            ptx::fence_proxy_async_generic_sync_restrict(ptx::sem_acquire, ptx::space_cluster, ptx::scope_cluster);

            cg::invoke_one(cg::coalesced_threads(), [&](){ptx::clusterlaunchcontrol_try_cancel_multicast(&result, &bar);});
        }

        // 每个 threadblock 各自追踪取消完成：
        if (cg::thread_block::thread_rank() == 0)
            ptx::mbarrier_arrive_expect_tx(ptx::sem_relaxed, ptx::scope_cluster, ptx::space_shared, &bar, sizeof(uint4));

        // 计算：
        int i = bx * blockDim.x + threadIdx.x;
        if (i < n)
            data[i] *= alpha;

        // 取消请求同步：
        while (!ptx::mbarrier_try_wait_parity(ptx::sem_acquire, ptx::scope_cluster, &bar, phase))
        {}
        phase ^= 1;

        // 取消请求解码：
        bool success = ptx::clusterlaunchcontrol_query_cancel_is_canceled(result);
        if (!success)
            break;

        bx = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_x<int>(result);
        bx += cg::cluster_group::block_index().x; // 加上局部偏移。

        // 向 async proxy release 对 result 的读：
        ptx::fence_proxy_async_generic_sync_restrict(ptx::sem_release, ptx::space_shared, ptx::scope_cluster);
    }
}

// 启动: kernel_cluster_launch_control<<<(n + 1023) / 1024, 1024>>>(data, n);
```

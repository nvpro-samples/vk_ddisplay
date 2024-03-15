/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "vkdd.hpp"

namespace vkdd {
typedef uint32_t MemTypeIndex;

class VulkanMemoryPool
{
public:
  class Allocation;

  VulkanMemoryPool(vk::Device device, DeviceMask deviceMask, MemTypeIndex memTypeIdx, bool keepMapped, size_t minPageAllocationSize = 4 << 20);
  ~VulkanMemoryPool();

  Allocation alloc(size_t size, size_t alignment);

private:
  vk::Device                         m_device;
  DeviceMask                         m_deviceMask;
  MemTypeIndex                       m_memTypeIdx;
  bool                               m_keepMapped;
  size_t                             m_minPageAllocationSize;
  std::vector<struct PageAllocation> m_pageAllocations;
  std::mutex                         m_mtx;

  void free(Allocation const& alloc);
};

class VulkanMemoryPool::Allocation
{
public:
  Allocation();
  Allocation(Allocation&& other);
  ~Allocation();

  Allocation& operator=(Allocation&& other);

  void             free();
  vk::DeviceMemory devMem() const { return m_devMem; }
  size_t           devMemOffset() const { return m_devMemOffset; }
  void*            mappedMem() const { return m_mappedMem; }

private:
  friend class VulkanMemoryPool;

  VulkanMemoryPool* m_memPool;
  vk::DeviceMemory  m_devMem;
  size_t            m_devMemOffset;
  void*             m_mappedMem;
  size_t            m_size;

  Allocation(VulkanMemoryPool* memPool, vk::DeviceMemory devMem, size_t devMemOffset, void* mappedMem, size_t size);

  void clear();
};
}  // namespace vkdd

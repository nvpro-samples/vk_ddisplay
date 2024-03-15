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

#include "vulkan_memory_pool.hpp"

namespace vkdd {
struct Interval
{
  size_t m_begin;
  size_t m_end;

  bool operator<(Interval const& right) const;
  bool intersects(Interval const& other) const;
};

bool Interval::operator<(Interval const& right) const
{
  return m_begin != right.m_begin ? m_begin < right.m_begin : m_end < right.m_end;
}

bool Interval::intersects(Interval const& other) const
{
  return m_begin < other.m_end && other.m_begin < m_end;
}

struct PageAllocation
{
  vk::Device             m_device;
  vk::UniqueDeviceMemory m_devMem;
  void*                  m_mapped;
  size_t                 m_size;
  std::vector<Interval>  m_freeIntervals;

  PageAllocation(PageAllocation&& other);
  PageAllocation(vk::Device device, vk::UniqueDeviceMemory&& devMem, size_t size);
  ~PageAllocation();

  std::optional<Interval> requestInterval(size_t size, size_t alignment);
  void                    returnInterval(Interval interval);
};

VulkanMemoryPool::Allocation::Allocation()
    : Allocation(nullptr, nullptr, 0, nullptr, 0)
{
}

VulkanMemoryPool::Allocation::Allocation(Allocation&& other)
    : m_memPool(other.m_memPool)
    , m_devMem(other.m_devMem)
    , m_devMemOffset(other.m_devMemOffset)
    , m_mappedMem(other.m_mappedMem)
    , m_size(other.m_size)
{
  other.clear();
}

VulkanMemoryPool::Allocation::Allocation(VulkanMemoryPool* memPool, vk::DeviceMemory devMem, size_t devMemOffset, void* mappedMem, size_t size)
    : m_memPool(memPool)
    , m_devMem(devMem)
    , m_devMemOffset(devMemOffset)
    , m_mappedMem(mappedMem)
    , m_size(size)
{
}

VulkanMemoryPool::Allocation::~Allocation()
{
  this->free();
}

VulkanMemoryPool::Allocation& VulkanMemoryPool::Allocation::operator=(Allocation&& other)
{
  if(this != &other)
  {
    this->free();
    m_memPool      = other.m_memPool;
    m_devMem       = other.m_devMem;
    m_devMemOffset = other.m_devMemOffset;
    m_mappedMem    = other.m_mappedMem;
    m_size         = other.m_size;
    other.clear();
  }
  return *this;
}

void VulkanMemoryPool::Allocation::clear()
{
  m_memPool      = nullptr;
  m_devMem       = nullptr;
  m_devMemOffset = 0;
  m_mappedMem    = nullptr;
  m_size         = 0;
}

void VulkanMemoryPool::Allocation::free()
{
  if(m_memPool)
  {
    m_memPool->free(*this);
    this->clear();
  }
}

PageAllocation::PageAllocation(PageAllocation&& other)
    : m_device(other.m_device)
    , m_devMem(std::move(other.m_devMem))
    , m_mapped(other.m_mapped)
    , m_size(other.m_size)
    , m_freeIntervals(std::move(other.m_freeIntervals))
{
  other.m_device = nullptr;
  other.m_mapped = nullptr;
  other.m_size   = 0;
}

PageAllocation::PageAllocation(vk::Device device, vk::UniqueDeviceMemory&& devMem, size_t size)
    : m_device(device)
    , m_devMem(std::move(devMem))
    , m_mapped(nullptr)
    , m_size(size)
    , m_freeIntervals{{0, m_size}}
{
}

PageAllocation::~PageAllocation()
{
  assert(m_size == 0
         || (m_freeIntervals.size() == 1 && m_freeIntervals.front().m_begin == 0 && m_freeIntervals.front().m_end == m_size));
  if(m_devMem && m_mapped)
  {
    m_device.unmapMemory(m_devMem.get());
  }
}

VulkanMemoryPool::VulkanMemoryPool(vk::Device device, DeviceMask deviceMask, MemTypeIndex memTypeIdx, bool keepMapped, size_t minPageAllocationSize)
    : m_device(device)
    , m_deviceMask(deviceMask)
    , m_memTypeIdx(memTypeIdx)
    , m_keepMapped(keepMapped)
    , m_minPageAllocationSize(minPageAllocationSize)
{
}

VulkanMemoryPool::~VulkanMemoryPool() {}

VulkanMemoryPool::Allocation VulkanMemoryPool::alloc(size_t size, size_t alignment)
{
  std::lock_guard guard(m_mtx);
  for(PageAllocation& pageAllocation : m_pageAllocations)
  {
    std::optional<Interval> interval = pageAllocation.requestInterval(size, alignment);
    if(interval.has_value())
    {
      return {this, pageAllocation.m_devMem.get(), interval.value().m_begin,
              m_keepMapped ? (char*)pageAllocation.m_mapped + interval.value().m_begin : nullptr,
              interval.value().m_end - interval.value().m_begin};
    }
  }
  size_t                      pageSize = std::max(size, m_minPageAllocationSize);
  vk::MemoryAllocateInfo      allocateInfo(pageSize, m_memTypeIdx);
  vk::MemoryAllocateFlagsInfo allocateFlagsInfo(vk::MemoryAllocateFlagBits::eDeviceMask, m_deviceMask);
  if(m_deviceMask != 0)
  {
    allocateInfo.setPNext(&allocateFlagsInfo);
  }
  m_pageAllocations.emplace_back(m_device, m_device.allocateMemoryUnique(allocateInfo), pageSize);
  std::array<char const*, 4> units = {"", "Ki", "Mi", "Gi"};
  uint32_t unitIdx      = (uint32_t)std::min((size_t)std::floor(std::log2((double)pageSize) / 10.0f), units.size() - 1);
  float    displayValue = (double)pageSize / (double)(std::size_t(1) << (10 * unitIdx));
  LOGI("New %s memory allocation: %.2f %sB.\n", m_keepMapped ? "system" : "device", displayValue, units[unitIdx]);
  if(m_keepMapped)
  {
    m_pageAllocations.back().m_mapped = m_device.mapMemory(m_pageAllocations.back().m_devMem.get(), 0, pageSize);
  }
  vk::DeviceMemory devMem   = m_pageAllocations.back().m_devMem.get();
  Interval         interval = m_pageAllocations.back().requestInterval(size, alignment).value();
  return {this, devMem, interval.m_begin, m_keepMapped ? (char*)m_pageAllocations.back().m_mapped + interval.m_begin : nullptr,
          interval.m_end - interval.m_begin};
}

void VulkanMemoryPool::free(Allocation const& allocation)
{
  std::lock_guard guard(m_mtx);
  auto            findIt = std::find_if(m_pageAllocations.begin(), m_pageAllocations.end(),
                                        [&](PageAllocation const& pa) { return pa.m_devMem.get() == allocation.m_devMem; });
  assert(findIt != m_pageAllocations.end());
  findIt->returnInterval({allocation.m_devMemOffset, allocation.m_devMemOffset + allocation.m_size});
}

std::optional<Interval> PageAllocation::requestInterval(size_t size, size_t alignment)
{
  for(auto it = m_freeIntervals.begin(); it != m_freeIntervals.end(); ++it)
  {
    Interval& interval     = *it;
    size_t    alignedBegin = interval.m_begin + (alignment - interval.m_begin % alignment) % alignment;
    if(alignedBegin + size <= interval.m_end)
    {
      if(alignedBegin == interval.m_begin && alignedBegin + size == interval.m_end)
      {
        m_freeIntervals.erase(it);
      }
      else if(alignedBegin + size == interval.m_end)
      {
        interval.m_end = alignedBegin;
      }
      else
      {
        size_t prevIntervalBegin = interval.m_begin;
        interval.m_begin         = alignedBegin + size;
        if(prevIntervalBegin != alignedBegin)
        {
          m_freeIntervals.insert(it, {prevIntervalBegin, alignedBegin});
        }
      }
      return {{alignedBegin, alignedBegin + size}};
    }
  }
  return {};
}

void PageAllocation::returnInterval(Interval interval)
{
  if(interval.m_begin == interval.m_end)
  {
    return;
  }
  auto lbIt = std::lower_bound(m_freeIntervals.begin(), m_freeIntervals.end(), interval);
  if(lbIt == m_freeIntervals.end())
  {
    if(!m_freeIntervals.empty() && m_freeIntervals.back().m_end == lbIt->m_begin)
    {
      m_freeIntervals.back().m_end = lbIt->m_end;
    }
    else
    {
      m_freeIntervals.emplace_back(interval);
    }
  }
  else
  {
    assert(lbIt == m_freeIntervals.begin() || !(lbIt - 1)->intersects(interval));
    assert(!lbIt->intersects(interval));
    bool mergeWithPrev = lbIt != m_freeIntervals.begin() && (lbIt - 1)->m_end == interval.m_begin;
    bool mergeWithNext = interval.m_end == lbIt->m_begin;
    if(mergeWithPrev && mergeWithNext)
    {
      (lbIt - 1)->m_end = lbIt->m_end;
      m_freeIntervals.erase(lbIt);
    }
    else if(mergeWithPrev)
    {
      (lbIt - 1)->m_end = interval.m_end;
    }
    else if(mergeWithNext)
    {
      lbIt->m_begin = interval.m_begin;
    }
    else
    {
      m_freeIntervals.insert(lbIt, interval);
    }
  }
}
}  // namespace vkdd

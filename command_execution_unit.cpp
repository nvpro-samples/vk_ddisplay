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

#include "command_execution_unit.hpp"

#include "logical_device.hpp"

namespace vkdd {
struct CommandBufferPool
{
  vk::UniqueCommandPool                m_commandPool;
  std::vector<vk::UniqueCommandBuffer> m_commandBuffers;
  uint32_t                             m_nextCommandBufferIndex;
};

struct QueueFamilyIndexData
{
  vk::UniqueFence                                        m_syncFence;
  std::unordered_map<std::thread::id, CommandBufferPool> m_perThreadCommandBufferPools;
};

struct CommandBufferInfo
{
  vk::CommandBufferSubmitInfo          m_commandBufferInfo;
  std::vector<vk::SemaphoreSubmitInfo> m_waitSemaphoreInfos;
  std::vector<vk::SemaphoreSubmitInfo> m_signalSemaphoreInfos;
};

CommandExecutionUnit::CommandExecutionUnit(LogicalDevice& logicalDevice)
    : m_logicalDevice(logicalDevice)
{
}

CommandExecutionUnit::~CommandExecutionUnit() {}

vk::Result CommandExecutionUnit::waitForIdle()
{
  std::vector<vk::Fence> fences;
  for(auto const& it : m_library)
  {
    fences.emplace_back(it.second.m_syncFence.get());
  }
  if(fences.empty())
  {
    return vk::Result::eSuccess;
  }
  vk::Result result = m_logicalDevice.vkDevice().waitForFences(fences, true, std::numeric_limits<uint64_t>::max());
  m_logicalDevice.vkDevice().resetFences(fences);
  return result;
}

void CommandExecutionUnit::waitForIdleAndReset()
{
  this->waitForIdle();
  for(auto const& queueIt : m_library)
  {
    for(auto const& threadIt : queueIt.second.m_perThreadCommandBufferPools)
    {
      m_logicalDevice.vkDevice().resetCommandPool(threadIt.second.m_commandPool.get());
    }
  }
}

std::vector<vk::CommandBuffer> CommandExecutionUnit::requestCommandBuffers(std::vector<uint32_t>     queueFamilyIndices,
                                                                           std::optional<DeviceMask> deviceMask)
{
  std::vector<vk::CommandBuffer> cmdBuffers;
  if(!queueFamilyIndices.empty())
  {
    std::lock_guard guard(m_mutex);
    for(uint32_t queueFamilyIndex : queueFamilyIndices)
    {
      cmdBuffers.emplace_back(this->requestCommandBufferUnguarded(queueFamilyIndex, deviceMask));
    }
  }
  return cmdBuffers;
}
vk::CommandBuffer CommandExecutionUnit::requestCommandBuffer(uint32_t queueFamilyIndex, std::optional<DeviceMask> deviceMask)
{
  std::lock_guard guard(m_mutex);
  return this->requestCommandBufferUnguarded(queueFamilyIndex, deviceMask);
}

vk::CommandBuffer CommandExecutionUnit::requestCommandBufferUnguarded(uint32_t queueFamilyIndex, std::optional<DeviceMask> deviceMask)
{
  CommandBufferPool* cbp = nullptr;
  if(auto findQueueIt = m_library.find(queueFamilyIndex); findQueueIt != m_library.end())
  {
    if(auto findThreadIt = findQueueIt->second.m_perThreadCommandBufferPools.find(std::this_thread::get_id());
       findThreadIt != findQueueIt->second.m_perThreadCommandBufferPools.end())
    {
      cbp = &findThreadIt->second;
    }
  }
  if(!cbp)
  {
    m_library[queueFamilyIndex].m_syncFence = m_logicalDevice.vkDevice().createFenceUnique({});
    cbp = &(m_library[queueFamilyIndex].m_perThreadCommandBufferPools[std::this_thread::get_id()] = {
                m_logicalDevice.vkDevice().createCommandPoolUnique({{}, queueFamilyIndex})});
  }
  if(cbp->m_commandBuffers.size() <= cbp->m_nextCommandBufferIndex)
  {
    vk::CommandBufferAllocateInfo graphicsCommandBufferAllocateInfo(cbp->m_commandPool.get(), vk::CommandBufferLevel::ePrimary, 1);
    cbp->m_commandBuffers.emplace_back(
        std::move(m_logicalDevice.vkDevice().allocateCommandBuffersUnique(graphicsCommandBufferAllocateInfo)[0]));
  }
  vk::CommandBuffer cmdBuffer     = cbp->m_commandBuffers[cbp->m_nextCommandBufferIndex++].get();
  m_commandBufferInfos[cmdBuffer] = {{cmdBuffer, deviceMask.value_or(DeviceMask())}, {}, {}};
  m_submitOrder[queueFamilyIndex].emplace_back(cmdBuffer);
  return cmdBuffer;
}

void CommandExecutionUnit::pushWaits(vk::CommandBuffer cmdBuffer, std::vector<vk::SemaphoreSubmitInfo> const& waitSemaphoreInfos)
{
  if(!waitSemaphoreInfos.empty())
  {
    std::lock_guard guard(m_mutex);
    auto            findIt = m_commandBufferInfos.find(cmdBuffer);
    if(findIt == m_commandBufferInfos.end())
    {
      LOGE("Unknown command buffer given.\n");
      return;
    }
    findIt->second.m_waitSemaphoreInfos.insert(findIt->second.m_waitSemaphoreInfos.end(), waitSemaphoreInfos.begin(),
                                               waitSemaphoreInfos.end());
  }
}

void CommandExecutionUnit::pushWait(vk::CommandBuffer cmdBuffer, vk::SemaphoreSubmitInfo waitSemaphoreInfo)
{
  this->pushWaits(cmdBuffer, {waitSemaphoreInfo});
}

void CommandExecutionUnit::pushSignals(vk::CommandBuffer cmdBuffer, std::vector<vk::SemaphoreSubmitInfo> const& signalSemaphoreInfos)
{
  if(!signalSemaphoreInfos.empty())
  {
    std::lock_guard guard(m_mutex);
    auto            findIt = m_commandBufferInfos.find(cmdBuffer);
    if(findIt == m_commandBufferInfos.end())
    {
      LOGE("Unknown command buffer given.\n");
      return;
    }
    findIt->second.m_signalSemaphoreInfos.insert(findIt->second.m_signalSemaphoreInfos.end(),
                                                 signalSemaphoreInfos.begin(), signalSemaphoreInfos.end());
  }
}

void CommandExecutionUnit::pushSignal(vk::CommandBuffer cmdBuffer, vk::SemaphoreSubmitInfo signalSemaphoreInfo)
{
  this->pushSignals(cmdBuffer, {signalSemaphoreInfo});
}

void CommandExecutionUnit::submit()
{
  for(auto const& it : m_submitOrder)
  {
    vk::Queue queue = m_logicalDevice.getQueue(it.first);
    // for(uint32_t i = 0; i < it.second.size(); ++i)
    // {
    //   CommandBufferInfo const& info = m_commandBufferInfos[it.second[i]];
    //   vk::SubmitInfo2 submit({}, info.m_waitSemaphoreInfos, info.m_commandBufferInfo, info.m_signalSemaphoreInfos);
    //   LOGI("%d\n", i);
    //   queue.submit2(submit, it.first == m_syncFenceQueueFamilyIndex && i == it.second.size() - 1 ? m_syncFence.get() : nullptr);
    // }
    std::vector<vk::SubmitInfo2> submits;
    for(vk::CommandBuffer cmdBuffer : it.second)
    {
      CommandBufferInfo const& info = m_commandBufferInfos[cmdBuffer];
      submits.emplace_back(vk::SubmitInfo2({}, info.m_waitSemaphoreInfos, info.m_commandBufferInfo, info.m_signalSemaphoreInfos));
    }
    queue.submit2(submits, m_library[it.first].m_syncFence.get());
  }
  m_submitOrder.clear();
  m_commandBufferInfos.clear();
}
}  // namespace vkdd

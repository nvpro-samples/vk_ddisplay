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
class VulkanMemoryObjectUploader
{
public:
  VulkanMemoryObjectUploader(class LogicalDevice& logicalDevice);
  ~VulkanMemoryObjectUploader();

  void memcpyHost2Buffer(vk::Buffer dstBuffer, size_t dstBufferOffset, void const* srcData, size_t size, vk::PipelineStageFlags2 dstStageMask);
  void          prepare(class CommandExecutionUnit& cmdExecUnit);
  void          finish();
  vk::Semaphore getSyncSemaphore() const { return m_syncSem.get(); }

private:
  LogicalDevice&                 m_logicalDevice;
  std::mutex                     m_mutex;
  std::vector<struct BufferCopy> m_bufferCopies;
  vk::UniqueSemaphore            m_syncSem;
  vk::CommandBuffer              m_transferCmdBuffer;
  vk::CommandBuffer              m_graphicsCmdBuffer;
};
}  // namespace vkdd

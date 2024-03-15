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

#version 450

layout(push_constant) uniform GlobalData
{
  mat4x4 m_view;
  mat4x4 m_proj;
  float  m_runtimeMillis;
}
g_data;

layout(location = 0) in vec3 vPos;
layout(location = 1) in vec3 vNormal;
layout(location = 2) in vec2 vTex;
layout(location = 3) in mat4 iModel;
layout(location = 7) in mat4 iInvModel;
layout(location = 11) in uint iUniqueId;
layout(location = 12) in float iShellHeight;
layout(location = 13) in float iExtrusion;

layout(location = 0) out vec3 fPos;
layout(location = 1) out vec3 fNormal;
layout(location = 2) out vec2 fTex;
layout(location = 3) out uint fUniqueId;
layout(location = 4) out float fShellHeight;

void main()
{
  vec4 worldPos = iModel * vec4(vPos + iExtrusion * normalize(vNormal), 1.0f);
  gl_Position   = g_data.m_proj * g_data.m_view * worldPos;
  fPos          = worldPos.xyz / worldPos.w;
  fNormal       = (vec4(vNormal, 0.0f) * iInvModel).xyz;
  fTex          = vTex;
  fShellHeight  = iShellHeight;
  fUniqueId     = iUniqueId;
}
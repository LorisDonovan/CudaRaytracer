#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <cassert>


// ----------Shader----------------------------------------
uint32_t CompileShader(uint32_t type, const char* shaderSrc);
uint32_t CreateShader(const char* vertexShader, const char* fragShader);
void ShaderBind(uint32_t shaderID);
void ShaderUnbind();
void ShaderSetInt(uint32_t shaderID, const std::string& name, int32_t value);

// ----------Framebuffer Quad------------------------------
void InitFbQuad(uint32_t& quadVA, uint32_t& quadVB, uint32_t& shaderID);

// ----------Texture---------------------------------------
uint32_t InitGLTexture(uint32_t width, uint32_t height);
void TextureBind(uint32_t textureID, uint32_t slot = 0);

// ----------Free up memory--------------------------------
void Cleanup(uint32_t& quadVao, uint32_t& quadVbo, uint32_t& textureID, uint32_t& shaderID);

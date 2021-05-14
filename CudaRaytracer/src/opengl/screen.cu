#include "screen.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>


// ----------Shader----------------------------------------
uint32_t CompileShader(uint32_t type, const char* shaderSrc)
{
	uint32_t id = glCreateShader(type);
	glShaderSource(id, 1, &shaderSrc, nullptr);
	glCompileShader(id);

	// Error Handling
	int32_t result;
	glGetShaderiv(id, GL_COMPILE_STATUS, &result);
	if (result == GL_FALSE)
	{
		GLint maxLength = 0;
		glGetShaderiv(id, GL_INFO_LOG_LENGTH, &maxLength);

		std::vector<GLchar> infoLog(maxLength);
		glGetShaderInfoLog(id, maxLength, &maxLength, &infoLog[0]);

		glDeleteShader(id);

		if (type == GL_VERTEX_SHADER)
			std::cerr << "Failed To Compile Vertex Shader!" << std::endl;
		if (type == GL_FRAGMENT_SHADER)
			std::cerr << "Failed To Compile Fragment Shader!" << std::endl;
		std::cout << infoLog.data() << std::endl;
		assert(false);
	}

	return id;
}

uint32_t CreateShader(const char* vertexShader, const char* fragShader)
{
	uint32_t program = glCreateProgram();
	uint32_t vs = CompileShader(GL_VERTEX_SHADER, vertexShader);
	uint32_t fs = CompileShader(GL_FRAGMENT_SHADER, fragShader);

	// linking
	glAttachShader(program, vs);
	glAttachShader(program, fs);
	glLinkProgram(program);
	glValidateProgram(program);

	// delete the intermediates
	glDeleteShader(vs);
	glDeleteShader(fs);

	return program;
}

void ShaderBind(uint32_t shaderID)
{
	glUseProgram(shaderID);
}

void ShaderUnbind()
{
	glUseProgram(0);
}

void ShaderSetInt(uint32_t shaderID, const std::string& name, int32_t value)
{
	GLint location = glGetUniformLocation(shaderID, name.c_str());
	glUniform1i(location, value);
}


// ---------Framebuffer Quad-------------------------------
void InitFbQuad(uint32_t& quadVA, uint32_t& quadVB, uint32_t& shaderID)
{
	float quadVertices[] = {
		// positions   // texCoords
		-1.0f,  1.0f,  0.0f, 1.0f,
		-1.0f, -1.0f,  0.0f, 0.0f,
		 1.0f, -1.0f,  1.0f, 0.0f,

		-1.0f,  1.0f,  0.0f, 1.0f,
		 1.0f, -1.0f,  1.0f, 0.0f,
		 1.0f,  1.0f,  1.0f, 1.0f
	};

	glGenBuffers(1, &quadVB);
	glGenVertexArrays(1, &quadVA);

	glBindVertexArray(quadVA);
	glBindBuffer(GL_ARRAY_BUFFER, quadVB);
	glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
	// Position Attribute
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	// Texture coordinate Attribute
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
	glEnableVertexAttribArray(1);

	// Shader
	const char* vertexShader = R"(
		#version 450 core

		layout(location = 0) in vec2 a_Position;
		layout(location = 1) in vec2 a_TexCoords;

		out vec2 TexCoords;

		void main()
		{
			TexCoords   = a_TexCoords;
			gl_Position = vec4(a_Position, 0.0f, 1.0f);
		}
	)";

	const char* fragShader = R"(
		#version 450 core

		in  vec2 TexCoords;
		out vec4 FragColor;

		uniform sampler2D u_Texture;

		void main()
		{
			vec3 col  = texture(u_Texture, TexCoords).rgb; 
			FragColor = vec4(col, 1.0f);
			//FragColor = vec4(TexCoords, 0.0f, 1.0f);
		}
	)";

	shaderID = CreateShader(vertexShader, fragShader);
	ShaderBind(shaderID);
	ShaderSetInt(shaderID, "u_Texture", 0);
	ShaderUnbind();
}


// ----------Texture---------------------------------------
uint32_t InitGLTexture(uint32_t width, uint32_t height)
{
	uint32_t textureID;
	glGenTextures(1, &textureID);
	glBindTexture(GL_TEXTURE_2D, textureID);
	// Texture properties
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	// set texture image
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
	// Unbind texture
	glBindTexture(GL_TEXTURE_2D, 0);

	return textureID;
}

void TextureBind(uint32_t textureID, uint32_t slot)
{
	glActiveTexture(GL_TEXTURE0 + slot);
	glBindTexture(GL_TEXTURE_2D, textureID);
}


// ---------Free up memory---------------------------------
void Cleanup(uint32_t& quadVao, uint32_t& quadVbo, uint32_t& textureID, uint32_t& shaderID)
{
	glDeleteBuffers(1, &quadVbo);
	glDeleteVertexArrays(1, &quadVao);
	glDeleteTextures(1, &textureID);
	glDeleteProgram(shaderID);
}

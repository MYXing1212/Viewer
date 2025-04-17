#ifndef SHADER_H
#define SHADER_H

#ifdef USE_GLAD
#include<glad/glad.h>	// include glad to get all the required OpenGL headers
#elif defined USE_GLEW
#include<GL/glew.h>
#endif

#include<glm/glm.hpp>
#include<glm/gtc/matrix_transform.hpp>
#include<glm/gtc/type_ptr.hpp>

#include<string>
#include<fstream>
#include<sstream>
#include<iostream>

class Shader
{
public:
	// the program ID
	unsigned int ID;

	Shader(){}

	// constructor generates the shader on the fly 
	// -------------------------------------------------------------
	Shader(const char* vertexPath, const char* fragmentPath, const char* geometryPath = nullptr)
	{
		// 1. retrieve the vertex/fragment source code from filePath
		std::string vertexCode;
		std::string fragmentCode;
		std::string geometryCode;
		std::fstream vShaderFile;
		std::ifstream fShaderFile;
		std::ifstream gShaderFile;
		bool useGeoShader = (nullptr != geometryPath);
		//std::cout << "useGeoShader = " << useGeoShader << std::endl;
		// ensure ifstream objects can throw exceptions:
		vShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
		fShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
		if (useGeoShader)
			gShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);

		try
		{
			// open files
			vShaderFile.open(vertexPath);
			fShaderFile.open(fragmentPath);
			if (useGeoShader)
				gShaderFile.open(geometryPath);
			std::stringstream vShaderStream, fShaderStream, gShaderStream;
			// read file's buffer contents into streams
			vShaderStream << vShaderFile.rdbuf();
			fShaderStream << fShaderFile.rdbuf();
			if (useGeoShader)
				gShaderStream << gShaderFile.rdbuf();
			// close file handlers
			vShaderFile.close();
			fShaderFile.close();
			if (useGeoShader)
				gShaderFile.close();
			// convert stream into string
			vertexCode = vShaderStream.str();
			fragmentCode = fShaderStream.str();
			geometryCode = gShaderStream.str();
		}
		catch (std::ifstream::failure e)
		{
			std::cout << "ERROR::SHADER::FILE_NOT_SUCCESFULLY_READ" << std::endl;
		}
		const char* vShaderCode = vertexCode.c_str();
		const char* fShaderCode = fragmentCode.c_str();
		const char* gShaderCode = geometryCode.c_str();
		// 2. compile shaders
		// vertex shader

		unsigned int vertex = glCreateShader(GL_VERTEX_SHADER);
		glShaderSource(vertex, 1, &vShaderCode, NULL);
		glCompileShader(vertex);
		checkCompileErrors(vertex, "VERTEX");
		// fragment Shader
		unsigned int fragment = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(fragment, 1, &fShaderCode, NULL);
		glCompileShader(fragment);
		checkCompileErrors(fragment, "FRAGMENT");
		
		// geometry Shader
		unsigned int geometry;
		if (useGeoShader){
			geometry = glCreateShader(GL_GEOMETRY_SHADER);
			glShaderSource(geometry, 1, &gShaderCode, NULL);
			glCompileShader(geometry);
			checkCompileErrors(geometry, "GEOMETRY");
		}

		// shader Program
		ID = glCreateProgram();
		glAttachShader(ID, vertex);
		glAttachShader(ID, fragment);
		if (useGeoShader)
			glAttachShader(ID, geometry);
		glLinkProgram(ID);
		checkCompileErrors(ID, "PROGRAM");
		// delete the shaders as they're linked into our program now and no longer necessary
		glDeleteShader(vertex);
		glDeleteShader(fragment);
		if (useGeoShader)
			glDeleteShader(geometry);
	}

	void init(const char* vertexPath, const char* fragmentPath, const char* geometryPath = nullptr)
	{
		// 1. retrieve the vertex/fragment source code from filePath
		std::string vertexCode;
		std::string fragmentCode;
		std::string geometryCode;
		std::fstream vShaderFile;
		std::ifstream fShaderFile;
		std::ifstream gShaderFile;
		bool useGeoShader = (nullptr != geometryPath);
		
		// ensure ifstream objects can throw exceptions:
		vShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
		fShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
		if (useGeoShader)
			gShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);

		try
		{
			// open files
			vShaderFile.open(vertexPath);
			fShaderFile.open(fragmentPath);
			if (useGeoShader)
				gShaderFile.open(geometryPath);
			std::stringstream vShaderStream, fShaderStream, gShaderStream;
			// read file's buffer contents into streams
			vShaderStream << vShaderFile.rdbuf();
			fShaderStream << fShaderFile.rdbuf();
			if (useGeoShader)
				gShaderStream << gShaderFile.rdbuf();
			// close file handlers
			vShaderFile.close();
			fShaderFile.close();
			if (useGeoShader)
				gShaderFile.close();
			// convert stream into string
			vertexCode = vShaderStream.str();
			fragmentCode = fShaderStream.str();
			geometryCode = gShaderStream.str();
		}
		catch (std::ifstream::failure e)
		{
			std::cout << "ERROR::SHADER::FILE_NOT_SUCCESFULLY_READ" << std::endl;
		}
		const char* vShaderCode = vertexCode.c_str();
		const char* fShaderCode = fragmentCode.c_str();
		const char* gShaderCode = geometryCode.c_str();
		// 2. compile shaders
		// vertex shader

		unsigned int vertex = glCreateShader(GL_VERTEX_SHADER);
		glShaderSource(vertex, 1, &vShaderCode, NULL);
		glCompileShader(vertex);
		checkCompileErrors(vertex, "VERTEX");
		// fragment Shader
		unsigned int fragment = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(fragment, 1, &fShaderCode, NULL);
		glCompileShader(fragment);
		checkCompileErrors(fragment, "FRAGMENT");

		// geometry Shader
		unsigned int geometry;
		if (useGeoShader){
			geometry = glCreateShader(GL_GEOMETRY_SHADER);
			glShaderSource(geometry, 1, &gShaderCode, NULL);
			glCompileShader(geometry);
			checkCompileErrors(geometry, "GEOMETRY");
		}

		// shader Program
		ID = glCreateProgram();
		glAttachShader(ID, vertex);
		glAttachShader(ID, fragment);
		if (useGeoShader)
			glAttachShader(ID, geometry);
		glLinkProgram(ID);
		checkCompileErrors(ID, "PROGRAM");
		// delete the shaders as they're linked into our program now and no longer necessary
		glDeleteShader(vertex);
		glDeleteShader(fragment);
		if (useGeoShader)
			glDeleteShader(geometry);

		std::cout << "shader: " << std::string(vertexPath) << " compile Done!" << std::endl;
		std::cout << "	      " << std::string(fragmentPath) << " compile Done!" << std::endl;
	}

	// constructor generates the shader on the fly 
	// -------------------------------------------------------------
	Shader(const char* computePath)
	{
		loadComputeShader(computePath);
	}

	void loadComputeShader(const char* computePath)
	{
		std::string computeCode;
		std::fstream cShaderFile;
		// ensure ifstream objects can throw exceptions:
		cShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);

		try
		{
			// open files
			cShaderFile.open(computePath);
			std::stringstream cShaderStream;
			// read file's buffer contents into streams
			cShaderStream << cShaderFile.rdbuf();
			// close file handlers
			cShaderFile.close();
			// convert stream into string
			computeCode = cShaderStream.str();
		}
		catch (std::ifstream::failure e)
		{
			std::cout << "ERROR::SHADER::FILE_NOT_SUCCESFULLY_READ" << std::endl;
		}
		const char* cShaderCode = computeCode.c_str();
		// 2. compile shaders
		unsigned int compute = glCreateShader(GL_COMPUTE_SHADER);
		glShaderSource(compute, 1, &cShaderCode, NULL);
		glCompileShader(compute);
		checkCompileErrors(compute, "COMPUTE");

		// shader Program
		ID = glCreateProgram();
		glAttachShader(ID, compute);
		glLinkProgram(ID);
		checkCompileErrors(ID, "PROGRAM");
		// delete the shaders as they're linked into our program now and no longer necessary
		glDeleteShader(compute);
	}

	// use/activate the shader
	Shader &use()
	{
		enabled = true;
		glUseProgram(ID);
		return *this;
	}

	void disable()
	{
		enabled = false;
		glUseProgram(0);
	}

	const bool &isEnable() const
	{
		return enabled;
	}

	// utility uniform functions
	void setBool(const std::string &name, bool value) const
	{
		glUniform1i(glGetUniformLocation(ID, name.c_str()), (int)value);
	}
	void setInt(const std::string &name, int value) const
	{
		glUniform1i(glGetUniformLocation(ID, name.c_str()), value);
	}
	void setFloat(const std::string &name, float value) const
	{
		glUniform1f(glGetUniformLocation(ID, name.c_str()), value);
	}
	void setMat4(const std::string &name, glm::mat4 mat) const
	{
		glUniformMatrix4fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, glm::value_ptr(mat));
	}
	void setVec3(const std::string &name, GLfloat x, GLfloat y, GLfloat z) const
	{
		glUniform3fv(glGetUniformLocation(ID, name.c_str()), 1, glm::value_ptr(glm::vec3(x, y, z)));
	}
	void setVec3(const std::string &name, glm::vec3 value) const
	{
		GLint loc = getUniformLocation(name);
		//if (loc == -1)	return;
		glUniform3fv(loc, 1, glm::value_ptr(value));
	}
	void setVec2(const std::string &name, glm::vec2 value) const
	{
		glUniform2fv(glGetUniformLocation(ID, name.c_str()), 1, glm::value_ptr(value));
	}
	void setVec2(const std::string &name, GLfloat x, GLfloat y) const
	{
		glUniform2f(glGetUniformLocation(ID, name.c_str()), x, y);
	}
	void setVec4(const std::string &name, glm::vec4 value) const
	{
		GLint loc = getUniformLocation(name);
		//if (loc == -1)	return;
		glUniform4fv(loc, 1, glm::value_ptr(value));
	}
	void setVec4(const std::string &name, GLfloat x, GLfloat y, GLfloat z, GLfloat w) const
	{
		glUniform4f(glGetUniformLocation(ID, name.c_str()), x, y, z, w);
	}
private:
	bool enabled = false;


	GLint getUniformLocation(const std::string &name) const 
	{
		GLint loc = glGetUniformLocation(ID, name.c_str());
		//if(loc == -1) std::cout << "未定义uniform变量" << name << " "<<loc<<  std::endl;
		return loc;
	}
	// utility function for checking shader compilation/linking errors.
	// -------------------------------------------------------------------
	void checkCompileErrors(unsigned int shader, std::string type)
	{
		int success;
		char infoLog[1024];
		if (type != "PROGRAM")
		{
			glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
			if (!success)
			{
				glGetShaderInfoLog(shader, 1024, NULL, infoLog);
				std::cout << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n" << infoLog << "\n -- ----------------------------------------------------- --" << std::endl;
			}
		}
		else
		{
			glGetProgramiv(shader, GL_LINK_STATUS, &success);
			if (!success)
			{
				glGetProgramInfoLog(shader, 1024, NULL, infoLog);
				std::cout << "ERROR::PROGRAM_LINKING_ERROR of type: " << type << "\n" << infoLog << "\n -- ----------------------------------------------------- --" << std::endl;
			}
		}
	}
};

#endif
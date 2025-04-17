#if defined USE_PCH
#include"stdafx.h"
#endif

#include<iostream>

#include<ft2build.h>
#include FT_FREETYPE_H

#include "ElementGeoRenderer.h"
#include"resource_manager.h"

cv::Mat Point3fVecToMat(std::vector<cv::Point3f> pts)
{
	cv::Mat b = cv::Mat(pts);
	//cout << "b = " << b << endl;
	cv::Mat c = b.clone();
	cv::Mat result = c.reshape(1, (int)pts.size());

	result.convertTo(result, CV_64FC1);
	//= b.reshape(1, (int)pts.size());
	//cout << "c = " << c << endl;
	return result;
}

glm::vec3 rotateAVec_(glm::vec3 v, glm::vec3 axis, float angle)
{
	if (angle == 0)
		return v;
	axis = glm::normalize(axis);
	float a = axis.x;
	float b = axis.y;
	float c = axis.z;
	float cosA = cos(angle);
	float sinA = sin(angle);
	glm::mat3 R;

	R[0][0] = a*a + (1 - a*a)* cosA;
	R[1][0] = a*b*(1 - cosA) + c*sinA;
	R[2][0] = a*c*(1 - cosA) - b*sinA;

	R[0][1] = a*b*(1 - cosA) - c*sinA;
	R[1][1] = b*b + (1 - b*b)*cosA;
	R[2][1] = b*c*(1 - cosA) + a*sinA;

	R[0][2] = a*c*(1 - cosA) + b*sinA;
	R[1][2] = b*c*(1 - cosA) - a*sinA;
	R[2][2] = c*c + (1 - c*c)*cosA;
	return (glm::inverse(R)*v);
}

// 得到与法线正交的两个切向量
void getOrthogonalVec_(glm::vec3 input, glm::vec3& u, glm::vec3& v)
{
	glm::vec3 axis = glm::normalize(glm::cross(glm::vec3(0, 0, 1), input));
	float angle = acos(glm::dot(input, glm::vec3(0, 0, 1)));
	u = rotateAVec_(glm::vec3(1, 0, 0), axis, angle);
	v = rotateAVec_(glm::vec3(0, 1, 0), axis, angle);
}

// 给定一组点，给出点坐标值在x,y,z三个方向上的范围
void boundingBox_(glm::vec3 *pts, int cnt, float *xmin, float *xmax, float *ymin, float *ymax, float *zmin, float *zmax)
{
	*xmin = pts[0].x;
	*xmax = pts[0].x;
	*ymin = pts[0].y;
	*ymax = pts[0].y;
	*zmin = pts[0].z;
	*zmax = pts[0].z;
	for (int i = 0; i < cnt; i++)
	{
		if (*xmin > pts[i].x) *xmin = pts[i].x;
		if (*xmax < pts[i].x) *xmax = pts[i].x;
		if (*ymin > pts[i].y) *ymin = pts[i].y;
		if (*ymax < pts[i].y) *ymax = pts[i].y;
		if (*zmin > pts[i].z) *zmin = pts[i].z;
		if (*zmax < pts[i].z) *zmax = pts[i].z;
	}
}

// 计算球面点 利用球参数方程 Tips OpenGL
cv::Point3f getSpherePoint(float u, float v){
	float x = sin(CV_PI*v)*cos(CV_PI * 2.0 * u);
	float y = sin(CV_PI*v)*sin(CV_PI * 2.0 * u);
	float z = cos(CV_PI*v);
	return cv::Point3f(x, y, z);
}

ElementGeoRenderer::ElementGeoRenderer()
{
	
}

ElementGeoRenderer::~ElementGeoRenderer()
{
	if (VAO != -1)
	{
		glDeleteVertexArrays(1, &VAO);
		glDeleteBuffers(1, &VBO);
	}	
}


void ElementGeoRenderer::initSphereData(float radius)
{
	int num = 200;
	vector<cv::Point3f> sd;
	float ustep = 1 / (float)num, vstep = 1 / (float)num;
	double u = 0, v = 0;
	// 绘制上端三角形组
	for (int i = 0; i < num; i++){
		sd.push_back(getSpherePoint(0, 0));
		sd.push_back(getSpherePoint(u, vstep));
		sd.push_back(getSpherePoint(u + ustep, vstep));
		u += ustep;
	}

	// 绘制中间四边形组
	u = 0;
	for (int i = 1; i < num; i++)
	{
		for (int j = 0; j < num; j++){
			sd.push_back(getSpherePoint(u, v));
			sd.push_back(getSpherePoint(u + ustep, v));
			sd.push_back(getSpherePoint(u + ustep, v + vstep));
			sd.push_back(getSpherePoint(u + ustep, v + vstep));
			sd.push_back(getSpherePoint(u, v + vstep));
			sd.push_back(getSpherePoint(u, v));
			u += ustep;
		}
		v += vstep;
	}

	// 绘制下端三角形组
	u = 0;
	for (int i = 0; i < num; i++){
		sd.push_back(getSpherePoint(0, 1));
		sd.push_back(getSpherePoint(u, 1 - vstep));
		sd.push_back(getSpherePoint(u + ustep, 1 - vstep));
		u += ustep;
	}
	sphereData = Point3fVecToMat(sd);
	sphereData = sphereData * radius;
	R_sphere = radius;
	sphereData.convertTo(sphereData, CV_32FC1);

	//cout << "maxx = " << maxM<float>(sphereData.col(0)) << endl;
	//cout << "maxy = " << maxM<float>(sphereData.col(1)) << endl;
	cout << "spherData = " << sphereData.size() << endl;
}

void ElementGeoRenderer::initLineData()
{
	int num = 2000;
	std::vector<cv::Point3f> dat(num);
	for (int i = 0; i < 2000; i++)
	{
		dat[i].x = i / (float)num;
		dat[i].y = 0;
		dat[i].z = 0;
	}
	lineData = Point3fVecToMat(dat);
	lineData.convertTo(lineData, CV_32FC1);
}

void ElementGeoRenderer::initCircleData(float radius)
{
	vector<cv::Point3f> sd;
	int num = 200;
	float deltaAngle = 2 * CV_PI / (float)num;
	sd.push_back(cv::Point3f(0.0f, 0.0f, 0.0f));
	for (int i = 0; i < num; i++)
	{
		sd.push_back(cv::Point3f(cos(deltaAngle*i)*radius,
		sin(deltaAngle*i)*radius, 0.0f));
	}
	sd.push_back(cv::Point3f(radius,0.0f, 0.0f));
	circleData = Point3fVecToMat(sd);
	R_circle = radius;
	circleData.convertTo(circleData, CV_32FC1);
}

void ElementGeoRenderer::initCylinderData(float radius, float h)
{
	vector<cv::Point3f> sd;
	int num = 200;
	float deltaAngle = 2 * CV_PI / (float)num;
	sd.push_back(cv::Point3f(radius, 0.0f, h));
	sd.push_back(cv::Point3f(radius, 0.0f, 0));
	for (int i = num-1; i >=0; i--){
		sd.push_back(cv::Point3f(cos(deltaAngle*i)*radius,
		sin(deltaAngle*i)*radius, h));
		sd.push_back(cv::Point3f(cos(deltaAngle*i)*radius,
			sin(deltaAngle*i)*radius, 0));
	}

	cylinderData = Point3fVecToMat(sd);
	R_cylinder = radius;
	h_cylinder = h;
	cylinderData.convertTo(cylinderData, CV_32FC1);
}

void ElementGeoRenderer::setPrjViewMatrix(glm::mat4 projection, glm::mat4 view)
{
	shader.use();
	shader.setMat4("projection", projection);
	shader.setMat4("view", view);
	this->projection = projection;
	this->view = view;
}
void ElementGeoRenderer::setModelMatrix(glm::mat4 model, bool updateModel/* = true*/)
{
	shader.use();
	if(updateModel)
		this->model = model;
	shader.setMat4("model", model);
}

// 绘制单位球
void ElementGeoRenderer::RenderSphere(float radius, glm::vec3 pos, glm::vec3 color)
{
	shader.use();
	glm::mat4 modelOld = model;
	glm::mat4 m = glm::mat4();
	m = glm::translate(m, pos);
	m = glm::scale(m, glm::vec3(radius));
	model = model * m;
	shader.setMat4("model", model);

	shader.setVec3("color", color);
	int num = 200;
	glBindVertexArray(this->VAO);
	glBindBuffer(GL_ARRAY_BUFFER, this->VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float)*sphereData.total(), (float*)sphereData.data, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (void*)0);
	glDrawArrays(GL_TRIANGLES, 0, num * 3 * 2 + num*num*6);
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	shader.use();
	shader.setMat4("model", modelOld);
	model = modelOld;
}

// 半径为归一化值 在[-1,1]坐标系下 pos也在[-1,1]坐标系下
void ElementGeoRenderer::RenderSphere2D(float radius, glm::vec2 pos, glm::vec3 color)
{
	glm::mat4 prjM = projection;
	glm::mat4 viewM = view;
	glm::mat4 modelM = model;
	shader.use();
	glm::mat4 m = glm::mat4();
	m = glm::translate(m, glm::vec3(pos.x, pos.y, 0));
	m = glm::scale(m, glm::vec3(radius, radius * (float)screenWidth / (float)screenHeight, radius));
	shader.setMat4("projection", glm::mat4());
	shader.setMat4("view", glm::mat4());
	shader.setMat4("model", m);

	shader.setVec3("color", color);
	int num = 200;
	glBindVertexArray(this->VAO);
	glBindBuffer(GL_ARRAY_BUFFER, this->VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float)*sphereData.total(), (float*)sphereData.data, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (void*)0);
	glDrawArrays(GL_TRIANGLES, 0, num * 3 * 2 + num*num * 6);
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	shader.use();
	shader.setMat4("projection", prjM);
	shader.setMat4("view", viewM);
	shader.setMat4("model", modelM);
}

// 绘制圆
void ElementGeoRenderer::RenderCircle(float radius, glm::vec3 color, bool bFilled/* = true*/)
{
	if (R_circle != radius)
		initCircleData(radius);

	shader.use();
	shader.setVec3("color", color);
	int num = 200;
	glBindVertexArray(this->VAO);
	glBindBuffer(GL_ARRAY_BUFFER, this->VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float)*circleData.total(), (float*)circleData.data, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (void*)0);
	if (bFilled)
		glDrawArrays(GL_TRIANGLE_FAN, 0, num * 3 + 2);
	else
		glDrawArrays(GL_LINE_STRIP, 1, num+1);
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void ElementGeoRenderer::RenderCircle(float radius, glm::vec3 cen, glm::vec3 norm, glm::vec3 color, 
	bool bFilled/* = true*/, bool dashed/* = false*/)
{
	if (norm.z > 0)
		norm = -norm;
	shader.use();
	shader.setVec3("color", color);
	glm::mat4 modelOld = model;
	glm::mat4 tmpModel = glm::mat4();
	tmpModel = glm::translate(tmpModel, cen);
	glm::vec3 axis = glm::cross(glm::normalize(norm), glm::vec3(0, 0, 1.0f));
	if (glm::length(axis) > 1e-10)
	{
		float l = glm::length(axis);
		tmpModel = glm::rotate(tmpModel, (float)(asin(glm::length(axis))*180.0 / CV_PI), axis.x / l, axis.y / l, axis.z / l);
	}
	tmpModel = glm::scale(tmpModel, glm::vec3(radius, radius, radius));
	model = model * tmpModel;
	shader.setMat4("model", model);

	int num = 200;
	glBindVertexArray(this->VAO);
	glBindBuffer(GL_ARRAY_BUFFER, this->VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float)*circleData.total(), (float*)circleData.data, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (void*)0);
	if (bFilled)
		glDrawArrays(GL_TRIANGLE_FAN, 0, num * 3 + 2);
	else
	{
		if (dashed)
			glDrawArrays(GL_LINES, 1, num + 1);
		else
			glDrawArrays(GL_LINE_STRIP, 1, num + 1);
	}

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	model = modelOld;
	shader.setMat4("model", model);
}

// 准备圆柱数据
void ElementGeoRenderer::RenderCylinder(float radius, float h, glm::vec3 color)
{
	if (h_cylinder != h || radius != R_cylinder)
		initCylinderData(radius, h);

	RenderCircle(radius, color);	
	shader.use();
	shader.setVec3("color", color);
	int num = 200;
	glBindBuffer(GL_ARRAY_BUFFER, this->VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float)*cylinderData.total(), (float*)cylinderData.data, GL_STATIC_DRAW);
	glBindVertexArray(this->VAO);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (void*)0);
	//glDrawArrays(GL_POINTS, 0, num);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 2 * num + 2);
	//glDrawArrays(GL_LINES, 0, 2 * num + 2);
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glm::mat4 t_model = glm::mat4();
	t_model = glm::translate(t_model, glm::vec3(0.0f, 0.0f, h));
	t_model = model * t_model;
	setModelMatrix(t_model, false);
	RenderCircle(radius, color);
	setModelMatrix(model, false);
}

// 准备圆锥数据
void ElementGeoRenderer::initConeData(float radius, float h)
{
	vector<cv::Point3f> sd;
	int num = 200;
	float deltaAngle = 2 * CV_PI / (float)num;
	sd.push_back(cv::Point3f(0, 0, h));
	for (int i = 0; i < num; i++){
		sd.push_back(cv::Point3f(cos(deltaAngle*i)*radius,
			sin(deltaAngle*i)*radius, 0.0f));
	}
	sd.push_back(cv::Point3f(radius, 0, 0.0f));
	coneData = Point3fVecToMat(sd);
	R_cone = radius;
	h_cone = h;
	coneData.convertTo(coneData, CV_32FC1);
}


void ElementGeoRenderer::RenderPlane(glm::vec2 topleft, glm::vec2 size, glm::vec3 color)
{
	shader.use();
	shader.setBool("render2D", false);
	GLfloat data[] = 
	{
		topleft.x, topleft.y, 0,
		topleft.x + size.x, topleft.y, 0,
		topleft.x, topleft.y + size.y, 0, 
		topleft.x + size.x, topleft.y + size.y, 0
	};

	shader.use();
	shader.setVec3("color", color);
	glBindBuffer(GL_ARRAY_BUFFER, this->VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(data), data, GL_STATIC_DRAW);
	glBindVertexArray(this->VAO);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (void*)0);
	//glDrawArrays(GL_POINTS, 0, num);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	//glDrawArrays(GL_LINES, 0, 2 * num + 2);
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	//shader.setBool("render2D", false);
}

void ElementGeoRenderer::RenderPlane(glm::vec3 cen, glm::vec3 norm, glm::vec2 size, glm::vec3 color, float alpha/* = 1.0*/)
{
	glm::vec3 u, v;
	norm = glm::normalize(norm);
	getOrthogonalVec_(norm, u, v);
	RenderPlane(cen, u, v, size, color, alpha);
}

void ElementGeoRenderer::RenderPlane(glm::vec3 cen, glm::vec3 u, glm::vec3 v, glm::vec2 size, glm::vec3 color, float alpha/* = 1.0*/)
{
	u = glm::normalize(u);
	v = glm::normalize(v);
	glm::vec3 data[] = {
		cen - u * size.x / 2.0f - v*size.y / 2.0f,
		cen - u * size.x / 2.0f + v*size.y / 2.0f,
		cen + u * size.x / 2.0f - v*size.y / 2.0f,
		cen + u * size.x / 2.0f + v*size.y / 2.0f
	};

	shader.use();
	shader.setVec3("color", color);
	shader.setFloat("alpha", alpha);
	glBindBuffer(GL_ARRAY_BUFFER, this->VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(data), data, GL_STATIC_DRAW);
	glBindVertexArray(this->VAO);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (void*)0);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	shader.setFloat("alpha", 1.0);
}

// 空间小面元的4个点顺次连接成一个小面片
void ElementGeoRenderer::RenderPlane(glm::vec3 A, glm::vec3 B, glm::vec3 C, glm::vec3 D, glm::vec3 color, float alpha/* = 1.0f*/)
{
	glm::vec3 data[] = {A, B, C, D};

	shader.use();
	shader.setVec3("color", color);
	shader.setFloat("alpha", alpha);
	glBindBuffer(GL_ARRAY_BUFFER, this->VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(data), data, GL_STATIC_DRAW);
	glBindVertexArray(this->VAO);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (void*)0);
	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	shader.setFloat("alpha", 1.0);
}

void ElementGeoRenderer::RenderCone(float radius, float h, glm::vec3 color){
	if (R_cone != radius || h != h_cone)
		initConeData(radius, h);
	RenderCircle(radius, color);

	int num = 200;
	shader.use();
	shader.setVec3("color", color);
	glBindBuffer(GL_ARRAY_BUFFER, this->VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float)*coneData.total(), (float*)coneData.data, GL_STATIC_DRAW);
	glBindVertexArray(this->VAO);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (void*)0);
	glDrawArrays(GL_TRIANGLE_FAN, 0, num+2);
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	//printf("renderCone!\n");
}

// 以底面为xoy平面 底面中点为坐标原点
void ElementGeoRenderer::RenderCube(glm::vec3 pos, glm::vec3 size, glm::vec3 color, bool bFilled/* = true*/)
{
	vector<cv::Point3f> cubeData;
	if (bFilled)
	{
		// 底面
		cubeData.push_back(cv::Point3f(pos.x - size.x / 2.0f, pos.y - size.y / 2.0f, 0.0f));
		cubeData.push_back(cv::Point3f(pos.x - size.x / 2.0f, pos.y + size.y / 2.0f, 0.0f));
		cubeData.push_back(cv::Point3f(pos.x + size.x / 2.0f, pos.y - size.y / 2.0f, 0.0f));
		cubeData.push_back(cv::Point3f(pos.x + size.x / 2.0f, pos.y + size.y / 2.0f, 0.0f));
		// 右侧面
		cubeData.push_back(cv::Point3f(pos.x + size.x / 2.0f, pos.y - size.y / 2.0f, size.z));
		cubeData.push_back(cv::Point3f(pos.x + size.x / 2.0f, pos.y + size.y / 2.0f, size.z));
		// 上顶面
		cubeData.push_back(cv::Point3f(pos.x - size.x / 2.0f, pos.y - size.y / 2.0f, size.z));
		cubeData.push_back(cv::Point3f(pos.x - size.x / 2.0f, pos.y + size.y / 2.0f, size.z));
		// 左侧面
		cubeData.push_back(cv::Point3f(pos.x - size.x / 2.0f, pos.y - size.y / 2.0f, 0));
		cubeData.push_back(cv::Point3f(pos.x - size.x / 2.0f, pos.y + size.y / 2.0f, 0));
		// 剩下的再画一波
		// 前侧面
		cubeData.push_back(cv::Point3f(pos.x - size.x / 2.0f, pos.y - size.y / 2.0f, 0.0f));
		cubeData.push_back(cv::Point3f(pos.x - size.x / 2.0f, pos.y - size.y / 2.0f, size.z));
		cubeData.push_back(cv::Point3f(pos.x + size.x / 2.0f, pos.y - size.y / 2.0f, size.z));
		cubeData.push_back(cv::Point3f(pos.x + size.x / 2.0f, pos.y - size.y / 2.0f, size.z));
		cubeData.push_back(cv::Point3f(pos.x + size.x / 2.0f, pos.y - size.y / 2.0f, 0.0f));
		cubeData.push_back(cv::Point3f(pos.x - size.x / 2.0f, pos.y - size.y / 2.0f, 0.0f));
		// 后侧面
		cubeData.push_back(cv::Point3f(pos.x - size.x / 2.0f, pos.y + size.y / 2.0f, 0.0f));
		cubeData.push_back(cv::Point3f(pos.x - size.x / 2.0f, pos.y + size.y / 2.0f, size.z));
		cubeData.push_back(cv::Point3f(pos.x + size.x / 2.0f, pos.y + size.y / 2.0f, 0.0f));

		cubeData.push_back(cv::Point3f(pos.x + size.x / 2.0f, pos.y + size.y / 2.0f, 0.0f));
		cubeData.push_back(cv::Point3f(pos.x + size.x / 2.0f, pos.y + size.y / 2.0f, size.z));
		cubeData.push_back(cv::Point3f(pos.x - size.x / 2.0f, pos.y + size.y / 2.0f, size.z));
	}
	else 
	{
		// 第一波 GL_LINE_STRIP 10
		cubeData.push_back(cv::Point3f(pos.x - size.x / 2.0f, pos.y - size.y / 2.0f, 0.0f));
		cubeData.push_back(cv::Point3f(pos.x - size.x / 2.0f, pos.y + size.y / 2.0f, 0.0f));
		cubeData.push_back(cv::Point3f(pos.x + size.x / 2.0f, pos.y + size.y / 2.0f, 0.0f));
		cubeData.push_back(cv::Point3f(pos.x + size.x / 2.0f, pos.y - size.y / 2.0f, 0.0f));
		cubeData.push_back(cv::Point3f(pos.x - size.x / 2.0f, pos.y - size.y / 2.0f, 0.0f));
		cubeData.push_back(cv::Point3f(pos.x - size.x / 2.0f, pos.y - size.y / 2.0f, size.z));
		cubeData.push_back(cv::Point3f(pos.x - size.x / 2.0f, pos.y + size.y / 2.0f, size.z));
		cubeData.push_back(cv::Point3f(pos.x + size.x / 2.0f, pos.y + size.y / 2.0f, size.z));
		cubeData.push_back(cv::Point3f(pos.x + size.x / 2.0f, pos.y - size.y / 2.0f, size.z));
		cubeData.push_back(cv::Point3f(pos.x - size.x / 2.0f, pos.y - size.y / 2.0f, size.z));
		// 第二波 GL_LINES 6
		cubeData.push_back(cv::Point3f(pos.x + size.x / 2.0f, pos.y - size.y / 2.0f, 0.0f));
		cubeData.push_back(cv::Point3f(pos.x + size.x / 2.0f, pos.y - size.y / 2.0f, size.z));
		cubeData.push_back(cv::Point3f(pos.x + size.x / 2.0f, pos.y + size.y / 2.0f, 0.0f));
		cubeData.push_back(cv::Point3f(pos.x + size.x / 2.0f, pos.y + size.y / 2.0f, size.z));
		cubeData.push_back(cv::Point3f(pos.x - size.x / 2.0f, pos.y + size.y / 2.0f, 0.0f));
		cubeData.push_back(cv::Point3f(pos.x - size.x / 2.0f, pos.y + size.y / 2.0f, size.z));
	}
	cv::Mat data = Point3fVecToMat(cubeData);
	data.convertTo(data, CV_32FC1);
	shader.use();
	shader.setVec3("color", color);
	glBindBuffer(GL_ARRAY_BUFFER, this->VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float)*data.total(), (float*)data.data, GL_STATIC_DRAW);
	glBindVertexArray(this->VAO);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (void*)0);
	if (bFilled)
	{
		glDrawArrays(GL_TRIANGLE_STRIP, 0, 10);
		glDrawArrays(GL_TRIANGLES, 10, 12);
	}
	else
	{
		glDrawArrays(GL_LINE_STRIP, 0, 10);
		glDrawArrays(GL_LINES, 10, 6);
	}		
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void ElementGeoRenderer::RenderSquareColumn(glm::vec3 pos, float height, glm::vec2 size)
{
	bool enableDepthTesh = glIsEnabled(GL_DEPTH_TEST);
	glEnable(GL_DEPTH_TEST);
	vector<cv::Point3f> cubeData;

	// 底面
	cubeData.push_back(cv::Point3f(pos.x - size.x / 2.0f, pos.y - size.y / 2.0f, 0.0f));
	cubeData.push_back(cv::Point3f(pos.x - size.x / 2.0f, pos.y + size.y / 2.0f, 0.0f));
	cubeData.push_back(cv::Point3f(pos.x + size.x / 2.0f, pos.y - size.y / 2.0f, 0.0f));
	cubeData.push_back(cv::Point3f(pos.x + size.x / 2.0f, pos.y + size.y / 2.0f, 0.0f));
	// 右侧面
	cubeData.push_back(cv::Point3f(pos.x + size.x / 2.0f, pos.y - size.y / 2.0f, height));
	cubeData.push_back(cv::Point3f(pos.x + size.x / 2.0f, pos.y + size.y / 2.0f, height));
	// 上顶面
	cubeData.push_back(cv::Point3f(pos.x - size.x / 2.0f, pos.y - size.y / 2.0f, height));
	cubeData.push_back(cv::Point3f(pos.x - size.x / 2.0f, pos.y + size.y / 2.0f, height));
	// 左侧面
	cubeData.push_back(cv::Point3f(pos.x - size.x / 2.0f, pos.y - size.y / 2.0f, 0));
	cubeData.push_back(cv::Point3f(pos.x - size.x / 2.0f, pos.y + size.y / 2.0f, 0));
	// 剩下的再画一波
	// 前侧面
	cubeData.push_back(cv::Point3f(pos.x - size.x / 2.0f, pos.y - size.y / 2.0f, 0.0f));
	cubeData.push_back(cv::Point3f(pos.x - size.x / 2.0f, pos.y - size.y / 2.0f, height));
	cubeData.push_back(cv::Point3f(pos.x + size.x / 2.0f, pos.y - size.y / 2.0f, height));
	cubeData.push_back(cv::Point3f(pos.x + size.x / 2.0f, pos.y - size.y / 2.0f, height));
	cubeData.push_back(cv::Point3f(pos.x + size.x / 2.0f, pos.y - size.y / 2.0f, 0.0f));
	cubeData.push_back(cv::Point3f(pos.x - size.x / 2.0f, pos.y - size.y / 2.0f, 0.0f));
	// 后侧面
	cubeData.push_back(cv::Point3f(pos.x - size.x / 2.0f, pos.y + size.y / 2.0f, 0.0f));
	cubeData.push_back(cv::Point3f(pos.x - size.x / 2.0f, pos.y + size.y / 2.0f, height));
	cubeData.push_back(cv::Point3f(pos.x + size.x / 2.0f, pos.y + size.y / 2.0f, 0.0f));

	cubeData.push_back(cv::Point3f(pos.x + size.x / 2.0f, pos.y + size.y / 2.0f, 0.0f));
	cubeData.push_back(cv::Point3f(pos.x + size.x / 2.0f, pos.y + size.y / 2.0f, height));
	cubeData.push_back(cv::Point3f(pos.x - size.x / 2.0f, pos.y + size.y / 2.0f, height));

	cv::Mat data = Point3fVecToMat(cubeData);
	data.convertTo(data, CV_32FC1);
	shader.use();
	shader.setBool("renderColumn", true);
	glBindBuffer(GL_ARRAY_BUFFER, this->VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float)*data.total(), (float*)data.data, GL_STATIC_DRAW);
	glBindVertexArray(this->VAO);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (void*)0);

	glDrawArrays(GL_TRIANGLE_STRIP, 0, 10);
	glDrawArrays(GL_TRIANGLES, 10, 12);
	
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	shader.setBool("renderColumn", false);

	RenderCube(pos, glm::vec3(size.x + 0.001, size.y + 0.001, height + 0.001), glm::vec3(0.0f), false);
	if (!enableDepthTesh)
		glDisable(GL_DEPTH_TEST);
}

void ElementGeoRenderer::RenderBoundingBox(float xmin, float xmax, float ymin, float ymax, float zmin, 
	float zmax, glm::vec3 color)
{
	RenderCube(glm::vec3(0.5*(xmin + xmax), 0.5*(ymin + ymax), 0.5*(zmin + zmax)),
		glm::vec3(xmax - xmin, ymax - ymin, zmax - zmin), color, false);
}

void ElementGeoRenderer::RenderLine(glm::vec3 start, glm::vec3 end, glm::vec3 color)
{
	float ld[6] = { start.x, start.y, start.z, end.x, end.y, end.z };	
	shader.use();
	shader.setVec3("color", color);
	glBindBuffer(GL_ARRAY_BUFFER, this->VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float)*6, ld, GL_STATIC_DRAW);
	glBindVertexArray(this->VAO);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (void*)0);
	glDrawArrays(GL_LINE_STRIP, 0, 2);
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void ElementGeoRenderer::RenderHalfLine(glm::vec3 start, glm::vec3 dir, float length, glm::vec3 color)
{
	dir = glm::normalize(dir);
	float ld[6] = { start.x, start.y, start.z, start.x + dir.x * length, start.y + length * dir.y, start.z + dir.z * length };
	shader.use();
	shader.setVec3("color", color);
	glBindBuffer(GL_ARRAY_BUFFER, this->VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6, ld, GL_STATIC_DRAW);
	glBindVertexArray(this->VAO);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (void*)0);
	glDrawArrays(GL_LINE_STRIP, 0, 2);
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void ElementGeoRenderer::RenderLine2D(glm::vec2 start, glm::vec2 end, glm::vec3 color)
{
	float ld[] = { start.x, start.y, 0.0f, 
		end.x, end.y, 0.0f};
	shader.use();
	shader.setBool("render2D", true);
	shader.setVec3("color", color);
	glBindVertexArray(this->VAO);
	glBindBuffer(GL_ARRAY_BUFFER, this->VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(ld), ld, GL_STATIC_DRAW);
	glDrawArrays(GL_LINES, 0, 2);
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	shader.setBool("render2D", false);
	shader.disable();
}

void ElementGeoRenderer::RenderPts(glm::vec3 pts[], int cnt, const float &pointSize, glm::vec3 color)
{
	shader.use();
	shader.setVec3("color", color);
	shader.setFloat("pointSize", pointSize);
	//printf("cnt = %d\n", sizeof(pts));

	glBindVertexArray(this->VAO);
	glBindBuffer(GL_ARRAY_BUFFER, this->VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float)*3*cnt, pts, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (void*)0);
	glDrawArrays(GL_POINTS, 0, cnt);
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void ElementGeoRenderer::RenderPts(std::vector<glm::vec3> pts, const float &pointSize, glm::vec3 color,
	bool enableBlend/* = false*/)
{
	shader.use();
	shader.setVec3("color", color);
	shader.setFloat("pointSize", pointSize);
	//printf("cnt = %d\n", sizeof(pts));

	bool flag = glIsEnabled(GL_BLEND);
	if (enableBlend)
		glEnable(GL_BLEND);
	else
		glDisable(GL_BLEND);
	glBindVertexArray(this->VAO);
	glBindBuffer(GL_ARRAY_BUFFER, this->VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 3 * pts.size(), pts.data(), GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (void*)0);
	glDrawArrays(GL_POINTS, 0, pts.size());
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	if (flag)
		glEnable(GL_BLEND);
	else
		glDisable(GL_BLEND);
}

void ElementGeoRenderer::RenderPtsUseCross2D(std::vector<glm::vec2> pts, 
	std::vector<int> ptsFlags, 
	const float &len,
	const float &width, 
	glm::vec3 color, bool enableBlend/* = false*/)
{
	if (pts.empty())
		return;
	glDisable(GL_DEPTH_TEST);
	glm::vec2 xdir(1.0f, 0.0f);
	glm::vec2 ydir(0.0, 1.0f);
	for (int i = 0; i < pts.size(); i++)
	{
		if (ptsFlags[i] == 1)
		{
			RenderPlane2D(pts[i] + xdir*len / 2.0f, pts[i] - xdir*len / 2.0f, width, color);
			RenderPlane2D(pts[i] + ydir*len / 2.0f, pts[i] - ydir*len / 2.0f, width, color);
		}
		else if (ptsFlags[i] == 0)
		{
			RenderPlane2D(pts[i] + xdir*len / 2.0f, pts[i] - xdir*len / 2.0f, width, glm::vec3(1.0f, 0.0f, 0.0f));
			RenderPlane2D(pts[i] + ydir*len / 2.0f, pts[i] - ydir*len / 2.0f, width, glm::vec3(1.0f, 0.0f, 0.0f));
		}
	}
}

void ElementGeoRenderer::RenderPtsUseCross2D(std::vector<glm::vec2> pts, const float &len, 
	glm::vec3 color, bool enableBlend/* = false*/)
{
	if (pts.empty())
		return;

	glDisable(GL_DEPTH_TEST);
	pts2D_record.assign(pts.begin(), pts.end());
	float *ld = new float[pts.size() * 4 * 3];
	for (int i = 0; i < pts.size(); i++)
	{
		ld[i * 12 + 0] = pts[i].x + len;
		ld[i * 12 + 1] = pts[i].y;
		ld[i * 12 + 2] = 0;

		ld[i * 12 + 3] = pts[i].x - len;
		ld[i * 12 + 4] = pts[i].y;
		ld[i * 12 + 5] = 0;
		
		ld[i * 12 + 6] = pts[i].x;
		ld[i * 12 + 7] = pts[i].y - len;
		ld[i * 12 + 8] = 0;
		
		ld[i * 12 + 9] = pts[i].x;
		ld[i * 12 + 10] = pts[i].y + len;
		ld[i * 12 + 11] = 0;
		//printf("ld[%d] x = %f y = %f\n",i, pts[i].x, pts[i].y);
	}

	shader.use();
	shader.setVec3("color", color);
	shader.setBool("render2D", true);

	bool flag = glIsEnabled(GL_BLEND);
	if (enableBlend)
		glEnable(GL_BLEND);
	else
		glDisable(GL_BLEND);
	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 12 * pts.size(), ld, GL_STATIC_DRAW);
	//glEnableVertexAttribArray(0);
	//glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (void*)0);
	glDrawArrays(GL_LINES, 0, 4*pts.size());
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	if (flag)
		glEnable(GL_BLEND);
	else
		glDisable(GL_BLEND);
	shader.setBool("render2D", false);
	shader.disable();

	delete[]ld;
}

// 绘制二维环
void ElementGeoRenderer::RenderLoop2D(std::vector<glm::vec2> pts, glm::vec3 color, bool enableBlend/* = false*/)
{
	shader.use();
	shader.setVec3("color", color);
	shader.setBool("render2D", true);

	float *pd = new float[(pts.size()) * 3];
	for (int i = 0; i < pts.size(); i++)
	{
		pd[i * 3 + 0] = pts[i].x;
		pd[i * 3 + 1] = pts[i].y;
		pd[i * 3 + 2] = 0;
	}
	/*pd[pts.size() * 3 + 0] = pts[0].x;
	pd[pts.size() * 3 + 1] = pts[0].y;
	pd[pts.size() * 3 + 2] = 0;*/

	bool flag = glIsEnabled(GL_BLEND);
	if (enableBlend)
		glEnable(GL_BLEND);
	else
		glDisable(GL_BLEND);
	glBindVertexArray(this->VAO);
	glBindBuffer(GL_ARRAY_BUFFER, this->VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 3 * (pts.size()), pd, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (void*)0);
	glDrawArrays(GL_LINE_LOOP, 0, pts.size());
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	if (flag)
		glEnable(GL_BLEND);
	else
		glDisable(GL_BLEND);
	shader.setBool("render2D", false);
	delete[] pd;
}

// 绘制四棱锥台
cv::Vec6f ElementGeoRenderer::RenderFrustum(glm::vec3 start, glm::vec3 dir, float nearPlane,
	float farPlane, float angleWidth, float angleHeight, glm::vec3 color, float alpha/* = 1.0f*/)
{
	dir = dir / glm::length(dir);
	glm::vec3 u, v;
	getOrthogonalVec_(dir, u, v);

	float halfWidthNear = std::tan(angleWidth / 2.0) *nearPlane;
	float halfHeightNear = std::tan(angleHeight / 2.0) *nearPlane;
	float halfWidthFar = std::tan(angleWidth / 2.0) *farPlane;
	float halfHeightFar = std::tan(angleHeight / 2.0) *farPlane;

	glm::vec3 nearP1 = start + dir * nearPlane + u * halfWidthNear + v*halfHeightNear;
	glm::vec3 nearP2 = start + dir * nearPlane - u * halfWidthNear + v*halfHeightNear;
	glm::vec3 nearP3 = start + dir * nearPlane - u * halfWidthNear - v*halfHeightNear;
	glm::vec3 nearP4 = start + dir * nearPlane + u * halfWidthNear - v*halfHeightNear;

	glm::vec3 farP1 = start + dir * farPlane + u * halfWidthFar + v*halfHeightFar;
	glm::vec3 farP2 = start + dir * farPlane - u * halfWidthFar + v*halfHeightFar;
	glm::vec3 farP3 = start + dir * farPlane - u * halfWidthFar - v*halfHeightFar;
	glm::vec3 farP4 = start + dir * farPlane + u * halfWidthFar - v*halfHeightFar;

	float xmin = 0, xmax = 0, ymin = 0, ymax = 0, zmin = 0, zmax = 0;

	glm::vec3 frustumData[] = {
		// 前
		nearP1, nearP2, nearP3, 
		nearP3, nearP4, nearP1,
		// 后
		farP1, farP2, farP3, 
		farP3, farP4, farP1,
		// 上
		nearP1, nearP2, farP2, 
		farP2, farP1, nearP1,
		// 下
		nearP4, nearP3, farP3, 
		farP3, farP4, nearP4, 
		// 左
		nearP1, farP1, farP4, 
		farP4, nearP4, nearP1, 
		// 右
		nearP2, farP2, farP3, 
		farP3, nearP3, nearP2
	};
	boundingBox_(frustumData, 36, &xmin, &xmax, &ymin, &ymax, &zmin, &zmax);

	shader.use();
	shader.setVec3("color", color);
	shader.setFloat("alpha", alpha);

	bool flagBlend = glIsEnabled(GL_BLEND);
	glEnable(GL_BLEND);
	glBindBuffer(GL_ARRAY_BUFFER, this->VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(frustumData), frustumData, GL_STATIC_DRAW);
	glBindVertexArray(this->VAO);
	glDrawArrays(GL_TRIANGLES, 0, 36);
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	if (!flagBlend)
		glDisable(GL_BLEND);

	return cv::Vec6f(xmin, xmax, ymin, ymax, zmin, zmax);
}

glm::mat4 ElementGeoRenderer::getTransformMatrix()
{
	return projection * view * model;
}

// 八个点的顺序是 上顶面左上角点-右上角点-右下角点-左下角点  下底面左上角点-右上角点-右下角点-左下角点
void ElementGeoRenderer::RenderBoundingBox(std::vector<glm::vec3> vertex, glm::vec3 color)
{
	shader.use();
	shader.setVec3("color", color);

	bool flagBlend = glIsEnabled(GL_BLEND);
	glEnable(GL_BLEND);
	glBindBuffer(GL_ARRAY_BUFFER, this->VBO);

	// 绘制填充cube
	glm::vec3 boxData[] = {
		// 上面4条棱
		vertex[0], vertex[1],
		vertex[1], vertex[2],
		vertex[2], vertex[3],
		vertex[3], vertex[0],
		// 下面4条棱
		vertex[4], vertex[5],
		vertex[5], vertex[6],
		vertex[6], vertex[7],
		vertex[7], vertex[4],
		// 侧面4条棱
		vertex[0], vertex[4],
		vertex[1], vertex[5],
		vertex[2], vertex[6],
		vertex[3], vertex[7],
	};	
	glBufferData(GL_ARRAY_BUFFER, sizeof(boxData), boxData, GL_STATIC_DRAW);
	glBindVertexArray(this->VAO);
	glDrawArrays(GL_LINES, 0, 24);
	
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	if (!flagBlend)
		glDisable(GL_BLEND);
}

// 清空背景色 默认为黑色
void ElementGeoRenderer::clear(glm::vec3 bkColor/* = GLM_BLACK*/)
{
	shader.use();
	shader.setVec3("color", bkColor);
	shader.setBool("render2D", true);

	bool flagBlend = glIsEnabled(GL_BLEND);
	bool flagDepthTest = glIsEnabled(GL_DEPTH_TEST);
	glDisable(GL_BLEND);
	glDisable(GL_DEPTH_TEST);

	float vertices[] = { 
		1.0f, 1.0f, 0.0f,
		-1.0f, 1.0f, 0.0f,
		1.0f, -1.0f, 0.0f,
		
		-1.0f, 1.0f, 0.0f,
		-1.0f, -1.0f, 0.0f,
		1.0f, -1.0f, 0.0f,
	};

	glBindBuffer(GL_ARRAY_BUFFER, this->VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
	glBindVertexArray(this->VAO);
	glDrawArrays(GL_TRIANGLES, 0, 6);
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	if (flagBlend) glEnable(GL_BLEND);
	if (flagDepthTest) glEnable(GL_DEPTH_TEST);
	shader.setBool("render2D", false);
}

	// 给出起点 给出终点  给出条带宽度
void ElementGeoRenderer::RenderPlane2D(glm::vec2 start, glm::vec2 end, float width, glm::vec3 color, 
	bool bFilled/* = true*/)
{
	glm::vec2 dir = glm::normalize(end - start);
	glm::vec2 v(-dir.y, dir.x);
	v = v * width / 2.0f;

	shader.use();
	shader.setVec3("color", color);
	shader.setBool("render2D", true);

	bool flagBlend = glIsEnabled(GL_BLEND);
	bool flagDepthTest = glIsEnabled(GL_DEPTH_TEST);
	glDisable(GL_BLEND);
	glDisable(GL_DEPTH_TEST);

	if (bFilled)
	{
		float vertices[] = {
			start.x - v.x, start.y - v.y, 0.0f,
			start.x + v.x, start.y + v.y, 0.0f,
			end.x + v.x, end.y + v.y, 0.0f,

			end.x + v.x, end.y + v.y, 0.0f,
			end.x - v.x, end.y - v.y, 0.0f,
			start.x - v.x, start.y - v.y, 0.0f,
		};
		glBindBuffer(GL_ARRAY_BUFFER, this->VBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
		glBindVertexArray(this->VAO);
		glDrawArrays(GL_TRIANGLES, 0, 6);
	}
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	if (flagBlend) glEnable(GL_BLEND);
	if (flagDepthTest) glEnable(GL_DEPTH_TEST);
	shader.setBool("render2D", false);
}



void ElementGeoRenderer::RenderPlane2D(glm::vec2 tl, glm::vec2 br, glm::vec3 color, bool bFilled/* = true*/)
{
	shader.use();
	shader.setVec3("color", color);
	shader.setBool("render2D", true);
	shader.setBool("renderGraySeq", false);
	shader.setBool("enableDepthFilter", false);
	shader.setBool("renderColumn", false);
	shader.setBool("showTexture", true);
	
	bool flagBlend = glIsEnabled(GL_BLEND);
	bool flagDepthTest = glIsEnabled(GL_DEPTH_TEST);
	glDisable(GL_BLEND);
	glDisable(GL_DEPTH_TEST);

	if (bFilled)
	{
		float vertices[] = {
			tl.x, tl.y, 0.0f,
			br.x, br.y, 0.0f,
			br.x, tl.y, 0.0f,

			tl.x, tl.y, 0.0f,
			tl.x, br.y, 0.0f,
			br.x, br.y, 0.0f,
		};
		glBindBuffer(GL_ARRAY_BUFFER, this->VBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
		glBindVertexArray(this->VAO);
		glDrawArrays(GL_TRIANGLES, 0, 6);
	}
	else
	{
		float vertices[] = {
			tl.x, tl.y, 0.0f,
			br.x, tl.y, 0.0f,
			br.x, br.y, 0.0f,
			tl.x, br.y, 0.0f, 
			tl.x, tl.y, 0.0f
		};
		glBindBuffer(GL_ARRAY_BUFFER, this->VBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
		glBindVertexArray(this->VAO);
		glDrawArrays(GL_LINE_STRIP, 0, 5);
	}

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	if (flagBlend) glEnable(GL_BLEND);
	if (flagDepthTest) glEnable(GL_DEPTH_TEST);
	shader.setBool("render2D", false);
}

void ElementGeoRenderer::RenderBackground()
{
	shader.use();
	shader.setBool("showTexture", false);
	shader.setBool("enableLight", true);
	shader.setVec3("viewPos", 0.0f, 0.0f, 1.0f);
	// 设置光照
	shader.setVec3("pointLights[0].ambient", 0.5f, 0.5f, 0.5f);
	shader.setVec3("pointLights[0].diffuse", 1.0f, 1.0f, 1.0f);
	shader.setVec3("pointLights[0].specular", 1.0f, 1.0f, 1.0f);
	shader.setVec3("pointLights[0].position", 0.0f, 0.0f, 1.0f);
	// 设置材料
	shader.setVec4("material.ambient", 0.231250f, 0.231250f, 0.231250f, 1.0f);
	shader.setVec4("material.diffuse", 0.277500f, 0.277500f, 0.277500f, 1.0f);
	shader.setVec4("material.specular", 0.973911f, 0.973911f, 0.973911f, 1.0f);
	shader.setFloat("material.shininess", 40.599998f);

	glm::vec2 tl(-1.0f, 1.0f);
	glm::vec2 br(1.0f, -1.0f);

	float vertices[] = 
	{
		tl.x, tl.y, -1.0f,
		br.x, br.y, -1.0f,
		br.x, tl.y, -1.0f,

		tl.x, tl.y, -1.0f,
		tl.x, br.y, -1.0f,
		br.x, br.y, -1.0f,
	};
	shader.setBool("render2D", true);

	glBindBuffer(GL_ARRAY_BUFFER, this->VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
	glBindVertexArray(this->VAO);
	glDrawArrays(GL_TRIANGLES, 0, 6);

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	shader.setBool("render2D", false);
	shader.setBool("enableLight", false);
	shader.setBool("showTexture", true);
}

void ElementGeoRenderer::init(int screenWidth, int screenHeight)
{
	this->screenWidth = screenWidth;
	this->screenHeight = screenHeight;
	R_sphere = 0.0f;
	R_cylinder = 0.0f;
	R_circle = 0.0f;
	h_cylinder = 0.0f;
	R_cone = 0.0;
	h_cone = 0.0;

	shader = ResourceManager::LoadShader("shaders\\element_geo.vert", "shaders\\element_geo.frag", nullptr, "points");
	shader.use();
	shader.setMat4("projection", glm::mat4());
	shader.setMat4("view", glm::mat4());
	shader.setMat4("model", glm::mat4());
	shader.setVec3("cen", glm::vec3(0.0f));
	shader.setVec2("screenSize", screenWidth, screenHeight);

	// Configure VAO/VBO for texture quads
	glGenVertexArrays(1, &this->VAO);
	glGenBuffers(1, &this->VBO);
	glBindVertexArray(this->VAO);
	glBindBuffer(GL_ARRAY_BUFFER, this->VBO);
	//glBufferData(GL_ARRAY_BUFFER, sizeof(sphereData), (float*)sphereData, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (void*)0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	initSphereData(1.0f);
	initCircleData(1.0f);
	initLineData();
}

// 给出鼠标位置，返回归一化后的坐标点
glm::vec2 ElementGeoRenderer::getNormalizedCoord(const float &xpos, const float &ypos)
{
	return glm::vec2(
		xpos / (float)screenWidth * 2.0f - 1.0f,
		-1.0f * (ypos / (float)screenHeight * 2.0f - 1.0f)
		);
}

void ElementGeoRenderer::updateROI(const glm::vec2 &pt)
{
	if (isSelectingROI)
	{
		float x = std::min(std::max(pt.x, -1.0f), 1.0f);
		float y = std::min(std::max(pt.y, -1.0f), 1.0f);
		RenderPlane2D(roiStart, glm::vec2(x, y), glm::vec3(0.0,0.0, 1.0f), false);
	}
}

void ElementGeoRenderer::endSelectROI(const glm::vec2 &endPt)
{
	isSelectingROI = false;
	float x = std::min(std::max(endPt.x, -1.0f), 1.0f);
	float y = std::min(std::max(endPt.y, -1.0f), 1.0f);
	roiEnd = glm::vec2(x, y);
}

void ElementGeoRenderer::startSelectROI(const glm::vec2 &startPt)
{
	isSelectingROI = true;
	roiStart = startPt;
}

void ElementGeoRenderer::RenderGraySeq(const std::vector<float> grays, glm::vec3 color,
	const float &lowThresh/* = 0*/, const float &highThresh/* = 255.0f*/)
{
	if (grays.empty())
		return;
	int N = (int)grays.size();
	if (N == 0)
		return;
	float *ld = new float[N * 2 * 3];
	for (int i = 0; i < N; i++)
	{
		ld[3 * i * 2 + 0] = -1.0 + (2.0 / N)*i;
		ld[3 * i * 2 + 1] = ((grays[i] - lowThresh) / highThresh)*2.0 - 1.0f;
		ld[3 * i * 2 + 2] = 0.0f;

		ld[3 * (2 * i + 1) + 0] = -1.0 + (2.0 / N)*(i + 1);
		ld[3 * (2 * i + 1) + 1] = ((grays[i] - lowThresh) / highThresh)*2.0 - 1.0f;
		ld[3 * (2 * i + 1) + 2] = 0.0f;
	}

	shader.use();
	shader.setBool("render2D", true);
	shader.setVec3("color", color);
	glBindBuffer(GL_ARRAY_BUFFER, this->VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * N * 2 * 3, ld, GL_STATIC_DRAW);
	glBindVertexArray(this->VAO);
	glDrawArrays(GL_LINE_STRIP, 0, N * 2);
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	shader.setBool("render2D", false);

	delete[] ld;
}

void ElementGeoRenderer::RenderGraySeq(const glm::vec2 &startPt, const glm::vec2 &endPt, glm::vec3 color,
	const float &lowThresh/* = 0*/, const float &highThresh/* = 255.0f*/)
{
	shader.use();
	shader.setVec2("startPt", startPt);
	shader.setVec2("endPt", endPt);
	shader.setVec2("grayRange", lowThresh, highThresh);
	shader.setBool("renderGraySeq", true);
	shader.setBool("render2D", true);
	shader.setVec3("color", color);
	glBindVertexArray(this->VAO);
	glBindBuffer(GL_ARRAY_BUFFER, this->VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * lineData.total(), (float*)lineData.data, GL_STATIC_DRAW);
	glDrawArrays(GL_LINE_STRIP, 0, lineData.rows);
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	shader.setBool("render2D", false);
	shader.setBool("renderGraySeq", false);
	shader.disable();
}

void ElementGeoRenderer::setGraySeqTexture(int GL_TEXTURE_id, int texture)
{
	glActiveTexture(GL_TEXTURE0+GL_TEXTURE_id);
	glBindTexture(GL_TEXTURE_2D, texture);
	shader.use();
	shader.setInt("graySeqTexture", GL_TEXTURE_id);
	shader.disable();
}



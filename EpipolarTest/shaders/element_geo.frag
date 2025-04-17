#version 450 core
out vec4 FragColor;

struct Material
{
	vec4 diffuse;
	vec4 specular;
	vec4 emission;
	float shininess;
};

struct DirLight
{
	vec3 direction;
	vec3 ambient;
	vec3 diffuse;
	vec3 specular;
};


struct PointLight 
{
	vec3 position;
	vec3 ambient;
	vec3 diffuse;
	vec3 specular;
};

struct SpotLight 
{
	vec3 position;
	vec3 direction;
	float cutOff;
	float outerCutOff;

	vec3 ambient;
	vec3 diffuse;
};

#define NR_POINT_LIGHTS 1

in vec3 position;
in vec3 posInViewSpace;

// 调试时使用
in vec3 vertColor;

uniform vec3 viewPos;
uniform DirLight dirLight;
uniform PointLight pointLights[NR_POINT_LIGHTS];
uniform SpotLight spotLight;
uniform Material material;

uniform bool enableDepthFilter = false;	// 使用深度
uniform bool showTexture = true;		// 显示纹理
uniform bool enableLight = false;		// 启用光照

uniform bool renderColumn=false;				// 渲染立柱 颜色渐变

uniform vec2 depthRange;				// [0] 最小值 [1]最大值

uniform vec2 screenSize;						// 屏幕尺寸

vec3 CalcPointLight(PointLight light, vec3 normal, vec3 fragPos, vec3 viewDir);

uniform vec3 color;
uniform float alpha = 1.0f;
uniform vec2 cen;
uniform bool multiColor=false;

float calcAlpha(vec2 pos){
	float dist = length(pos - cen);
	return clamp((25.0f - dist) / 25.0f, 0.0f, 1.0f);
}

vec3 calcColumnColor(float h)
{
	vec3 blue = vec3(0.0f, 0.0f, 1.0f);
	vec3 yellow = vec3(1.0f, 1.0f, 0.0f);
	return (blue + h / 100.0f * (yellow - blue));
}

void main()
{
	// 调试时使用
	//FragColor = vec4(vertColor, 1.0f);
	//return;

	//FragColor = vec4(color, 1.0f); 
	if(enableDepthFilter)
	{
		if(posInViewSpace.z < depthRange[0] || posInViewSpace.z > depthRange[1])
			discard;
	}

	// 如果是渲染立柱
	if(renderColumn)
	{
		FragColor = vec4(calcColumnColor(position.z), 1.0);
		return;
	}

	if(showTexture)
	{
		//FragColor = vec4(1.0f, 0.0f, 0.0f, 1.0f);
		if(cen.x >0)
			FragColor = vec4(color, calcAlpha(gl_FragCoord.xy));
		else 
			FragColor = vec4(color, alpha);
	}
	else
	{
		if(enableLight)
		{
			vec3 FragPos =  vec3(gl_FragCoord.x / screenSize.x, gl_FragCoord.y / screenSize.y, -1.0f);
			vec3 viewDir = normalize(viewPos - FragPos);
			vec3 result = CalcPointLight(pointLights[0], vec3(0.0f, 0.0f, 1.0f), FragPos, viewDir);
			FragColor = vec4(result, 1.0f);
			//FragColor = vec4(0.0f, 1.0f, 1.0f, 1.0f);
		}
	}	
}

vec3 CalcPointLight(PointLight light, vec3 normal, vec3 fragPos, vec3 viewDir)
{
	vec3 lightDir = normalize(light.position - fragPos);
	// diffuse shading
	float diff = max(dot(normal, lightDir), 0.0);
	// specular shading
	vec3 reflectDir = reflect(-lightDir, normal);
	float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);
	// attenuation
	// combine results
	vec3 ambient = light.ambient * material.diffuse.xyz * material.diffuse.w;
	vec3 diffuse = light.diffuse * diff * material.diffuse.xyz* material.diffuse.w;
	vec3 specular = light.specular * spec * material.specular.xyz * material.specular.w;
	//return vec3(gl_FragCoord.xy/800.0f, 1.0f);
	return (ambient + diffuse + specular);
}
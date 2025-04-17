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

uniform vec3 viewPos;
uniform DirLight dirLight;
uniform PointLight pointLights[NR_POINT_LIGHTS];
uniform SpotLight spotLight;
uniform Material material;

in vec4 Color;
in vec3 Normal;
in vec3 position;



uniform int nColorChannel;		// 使用nColorChannel的那个通道
uniform float near;
uniform float far;

uniform vec2 mousePos;

uniform float thresh;

uniform bool showTexture = true;		// 显示纹理
uniform bool enableLight = false;		// 启用光照

uniform vec3 pointsColor;

uniform sampler2D indexMap;

// function prototypes
vec3 CalcDirLight(DirLight light, vec3 normal, vec3 viewDir);
vec3 CalcPointLight(PointLight light, vec3 normal, vec3 fragPos, vec3 viewDir);
vec3 CalcSpotLight(SpotLight light, vec3 normal, vec3 fragPos, vec3 viewDir);

// 获取伪彩色 输入scalar 为0~1 float
vec3 getPseudoColor(float scalar)
{
	//scalar = (scalar + 1.0) / 2.0;
	vec3 color;
	if (scalar < 0.5f)
		color.x = 0.0f;
	else if (scalar < 0.75f)
		color.x = 4.0f * (scalar - 0.5f);
	else
		color.x = 1.0f;

	if (scalar < 0.25f)
		color.y = 4.0f * scalar;
	else if (scalar < 0.75f)
		color.y = 1.0f;
	else
		color.y = 1.0f - 4.0f * (scalar - 0.75f);

	if (scalar < 0.25f)
		color.z = 1.0f;
	else if (scalar < 0.5f)
		color.z = 1.0f - 4.0f * (scalar - 0.25f);
	else
		color.z = 0.0f;

	color = clamp(color, vec3(0.0), vec3(1.0));
	return color;
}

void main()
{
	if(position.z <= 0) 
		discard;	

	if(Color.w == -1.0f)
	{
		discard;
		return;
	}

	if(Color.w == 1.0f)
		FragColor = vec4(1.0f, 0.0f, 0.0f, 1.0f);
	else
	{
		float tmpc  = (Color.r - 0.025f) / 0.23f;

		if(showTexture)
			FragColor = Color;	
		else 
		{
			if(enableLight)
			{
				vec3 viewDir = normalize(viewPos - position);
				vec3 result = CalcPointLight(pointLights[0], vec3(0.0f), position, viewDir);
			//	FragColor = vec4(result, 1.0);
				FragColor = vec4(result, 1.0f);
			}
			else
				FragColor = vec4(getPseudoColor(tmpc), 1.0);
				//FragColor = vec4(pointsColor, 1.0);
		}
	}

	
	//float zspan = far - near;
	//FragColor = vec4(getPseudoColor((position.z - near) / zspan), 1.0f);
	//if(position.z < thresh)
	//	discard;

	//if(nColorChannel == 1)
	//	FragColor = vec4(getPseudoColor(color.x), 1.0f);
	//else if(nColorChannel == 3)
	//	FragColor = vec4(color.xyz, 1.0f);

	//float dist = (gl_FragCoord.x - mousePos.x) * (gl_FragCoord.x - mousePos.x) 
	//	+ (gl_FragCoord.y - mousePos.y) * (gl_FragCoord.y - mousePos.y);

	//dist = sqrt(dist);
	//if(dist < 5.0)
	//	FragColor = vec4(1.0f, 0.0f, 0.0f, 1.0f);
}

vec3 CalcDirLight(DirLight light, vec3 normal, vec3 viewDir)
{
	vec3 lightDir = normalize(-light.direction);
	// diffuse shading
	float diff = max(dot(normal, lightDir), 0.0);
	// specular shading
	vec3 reflectDir = reflect(-lightDir, normal);
	float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);
	// combine results
	vec3 ambient = light.ambient * vec3(1.0f)/*vec3(texture(material.diffuse, TexCoords))*/;
	vec3 diffuse = light.diffuse * diff * vec3(1.0f)/*vec3(texture(material.diffuse, TexCoords))*/;
	vec3 specular = light.specular  *spec * vec3(1.0f)/*vec3(texture(material.specular, TexCoords))*/;
	return (ambient + diffuse + specular);
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
	return (ambient + diffuse + specular);
}
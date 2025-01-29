#pragma once


#define shPI 3.1415926536f


#define sh2 float4
#define sh3 float3x3
// TODO sh3

sh2 sh2Zero()
{
	return float4(0.0f, 0.0f, 0.0f, 0.0f);
}

// Evaluates spherical harmonics basis for a direction dir.
// This follows [2] Appendix A2 order when storing in x, y, z and w.
// (evaluating the associated Legendre polynomials using the polynomial forms)
sh2 sh2Evaluate(float3 dir)
{
	sh2 result;
	result.x = 0.28209479177387814347403972578039f;			// L=0 , M= 0
	result.y =-0.48860251190291992158638462283836f * dir.y;	// L=1 , M=-1
	result.z = 0.48860251190291992158638462283836f * dir.z;	// L=1 , M= 0
	result.w =-0.48860251190291992158638462283836f * dir.x;	// L=1 , M= 1
	return result;
}

sh3 sh3Evaluate(float3 dir)
{
	sh3 result;
	result[0][0] = 0.28209479177387814347403972578039f;			// L=0 , M= 0
	result[0][1] =-0.48860251190291992158638462283836f * dir.y;	// L=1 , M=-1
	result[0][2] = 0.48860251190291992158638462283836f * dir.z;	// L=1 , M= 0
	result[1][0] =-0.48860251190291992158638462283836f * dir.x;	// L=1 , M= 1
	result[1][1] = 1.09254843059207907054338570580268f * dir.x * dir.y;						// L=2 , M=-2
	result[1][2] = 1.09254843059207907054338570580268f * dir.y * dir.z;  					// L=2 , M=-1
	result[2][0] = 0.31539156525252000603089369029571f * (3.0f * dir.z * dir.z - 1.0f);  	// L=2 , M= 0
	result[2][1] = 1.09254843059207907054338570580268f * dir.x * dir.z;  					// L=2 , M= 1
	result[2][2] = 0.54627421529603953527169285290134f * (dir.x * dir.x - dir.y * dir.y);  	// L=2 , M= 2

	return result;
}

// Recovers the value of a SH function in the direction dir.
float sh2Unproject(sh2 functionSh, float3 dir)
{
	sh2 sh = sh2Evaluate(dir);
	return dot(functionSh, sh);
}
float3 sh2Unproject(sh2 functionShX, sh2 functionShY, sh2 functionShZ, float3 dir)
{
	sh2 sh = sh2Evaluate(dir);
	return float3(dot(functionShX, sh), dot(functionShY, sh), dot(functionShZ, sh));
}

float matrix_dot(sh3 sh1, sh3 sh) {
	return sh1[0][0]*sh[0][0] + sh1[0][1]*sh[0][1] + sh1[0][2]*sh[0][2]
		 + sh1[1][0]*sh[1][0] + sh1[1][1]*sh[1][1] + sh1[1][2]*sh[1][2]
		 + sh1[2][0]*sh[2][0] + sh1[2][1]*sh[2][1] + sh1[2][2]*sh[2][2];
}

float sh3Unproject(sh3 functionSh, float3 dir)
{
	sh3 sh = sh3Evaluate(dir);
	return matrix_dot(functionSh, sh);
}


sh3 sh3TransformCosLobe(float3 normal)
{
	sh3 sh = sh3Evaluate(normal);
	sh[0][0] *= shPI;

	sh[0][1] *= 2.0943951023931954923f;	// L=1 , M=-1
	sh[0][2] *= 2.0943951023931954923f;	// L=1 , M= 0
	sh[1][0] *= 2.0943951023931954923f;	// L=1 , M= 1

	sh[1][1] *= 0.7853981633974483096f; 	// L=2 , M=-2
	sh[1][2] *= 0.7853981633974483096f;  	// L=2 , M=-1
	sh[2][0] *= 0.7853981633974483096f; 	// L=2 , M= 0
	sh[2][1] *= 0.7853981633974483096f;  	// L=2 , M= 1
	sh[2][2] *= 0.7853981633974483096f; 	// L=2 , M= 2

	return sh;
}

float3 sh3Unproject(sh3 functionShX, sh3 functionShY, sh3 functionShZ, float3 dir)
{
	sh3 sh = sh3Evaluate(dir);
	return float3(matrix_dot(functionShX, sh), matrix_dot(functionShY, sh), matrix_dot(functionShZ, sh));
}

// float4x4 M = float4x4(
// 	float4(c1*sh[2][2], c1*sh[1][1], c1*sh[2][1], c2*sh[1][0]),
// 	float4(c1*sh[1][1], -c1*sh[2][2], c1*sh[1][2], c2*sh[0][1]),
// 	float4(c1*sh[2][1], c1*sh[1][2], c3*sh[2][0], c2*sh[0][2]),
// 	float4(c2*sh[1][0], c2*sh[0][1], c2*sh[0][2], c4*sh[0][0] - c5*sh[2][0]),
// );

float3 sh3UnprojectCosLobe(sh3 ish[3], float3 normal)
{
	sh3 sh = sh3TransformCosLobe(normal);
	return float3(matrix_dot(ish[0], sh), matrix_dot(ish[1], sh), matrix_dot(ish[2], sh));
	// float c1 = 0.429043f; float c2 = 0.511664f; float c3 = 0.743125f; float c4 = 0.886227f; float c5 =  0.247708f;
	// float3 radiance;

	// [unroll]
	// for(int i = 0; i < 3; i++) {
	// 	sh3 sh = ish[i];

	// 	radiance[i] = c1*sh[2][2]*(normal.x * normal.x - normal.y * normal.y) + c3 * sh[2][0] * normal.z*normal.z + c4 * sh[0][0] - c5 * sh[2][0]
	// 		+ 2 * c1 * (sh[1][1]*normal.x*normal.y + sh[2][1] * normal.x * normal.z + sh[1][2] * normal.y * normal.z)
	// 		+ 2 * c2 * (sh[1][0]*normal.x + sh[0][1]*normal.y + sh[0][2]*normal.z);
	// }
	// return radiance;
}


// Adds two SH functions together.
sh2 sh2Add(sh2 shL, sh2 shR)
{
	return shL + shR;
}

// Scales a SH function uniformly by v.
sh2 sh2Scale(sh2 sh, float v)
{
	return sh * v;
}


// Adds two SH functions together.
sh3 sh3Add(sh3 shL, sh3 shR)
{
	return shL + shR;
}

// Scales a SH function uniformly by v.
sh3 sh3Scale(sh3 sh, float v)
{
	return sh * v;
}

float sh3FuncProductIntegral(sh3 shL, sh3 shR)
{
	return matrix_dot(shL, shR);
}
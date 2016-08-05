/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2011, Willow Garage, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include "device.hpp"
#include "device_launch_parameters.h"
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/gpu/gpu.hpp>
//using namespace cv;

using namespace pcl::device;
using namespace pcl::gpu;


namespace pcl
{
  namespace device
  {
    __global__ void
    computeVmapKernel (const PtrStepSz<unsigned short> depth, PtrStep<float> vmap, float fx_inv, float fy_inv, float cx, float cy)
    {
      int u = threadIdx.x + blockIdx.x * blockDim.x;
      int v = threadIdx.y + blockIdx.y * blockDim.y;

      if (u < depth.cols && v < depth.rows)
      {
        float z = depth.ptr (v)[u] / 1000.f; // load and convert: mm -> meters

        if (z != 0)
        {
          float vx = z * (u - cx) * fx_inv;
          float vy = z * (v - cy) * fy_inv;
          float vz = z;

          vmap.ptr (v                 )[u] = vx;
          vmap.ptr (v + depth.rows    )[u] = vy;
          vmap.ptr (v + depth.rows * 2)[u] = vz;
        }
        else
          vmap.ptr (v)[u] = numeric_limits<float>::quiet_NaN ();
	  
	  //printf(" All going well in ComputeVmapKernel ");

      }
    }

    __global__ void
    computeNmapKernel (int rows, int cols, const PtrStep<float> vmap, PtrStep<float> nmap)
    {
      int u = threadIdx.x + blockIdx.x * blockDim.x;
      int v = threadIdx.y + blockIdx.y * blockDim.y;

      if (u >= cols || v >= rows)
        return;

      if (u == cols - 1 || v == rows - 1)
      {
        nmap.ptr (v)[u] = numeric_limits<float>::quiet_NaN ();
        return;
      }

      float3 v00, v01, v10;
      v00.x = vmap.ptr (v  )[u];
      v01.x = vmap.ptr (v  )[u + 1];
      v10.x = vmap.ptr (v + 1)[u];

      if (!isnan (v00.x) && !isnan (v01.x) && !isnan (v10.x))
      {
        v00.y = vmap.ptr (v + rows)[u];
        v01.y = vmap.ptr (v + rows)[u + 1];
        v10.y = vmap.ptr (v + 1 + rows)[u];

        v00.z = vmap.ptr (v + 2 * rows)[u];
        v01.z = vmap.ptr (v + 2 * rows)[u + 1];
        v10.z = vmap.ptr (v + 1 + 2 * rows)[u];

        float3 r = normalized (cross (v01 - v00, v10 - v00));

        nmap.ptr (v       )[u] = r.x;
        nmap.ptr (v + rows)[u] = r.y;
        nmap.ptr (v + 2 * rows)[u] = r.z;
      }
      else
        nmap.ptr (v)[u] = numeric_limits<float>::quiet_NaN ();
	
	//printf(" All going well in ComputeNmapKernel ");
    }
    
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
   /* __global__ void
    computeRmapKernel(const PtrStep<float> vmap, PtrStep<float> nmap, const Mat33& Rmat, const float3& tvec, float fx, float fy, int rows, int cols, PtrStep<float> rmap)
    {
      int u = threadIdx.x + blockIdx.x * blockDim.x;
      int v = threadIdx.y + blockIdx.y * blockDim.y;
      
      const float qnan = pcl::device::numeric_limits<float>::quiet_NaN ();
      
      float3 v_temp, v_g = make_float3 (qnan, qnan, qnan);
      
      v_temp.x = vmap.ptr (v)[u];

        if (!isnan (v_temp.x))
        {
          v_temp.y = vmap.ptr (v + rows)[u];
          v_temp.z = vmap.ptr (v + 2 * rows)[u];

          v_g = Rmat * v_temp + tvec;

        }

        //normals
        float3 n_temp, n_g = make_float3 (qnan, qnan, qnan);
        n_temp.x = nmap.ptr (v)[u];

        if (!isnan (n_temp.x))
        {
          n_temp.y = nmap.ptr (v + rows)[u];
          n_temp.z = nmap.ptr (v + 2 * rows)[u];

          n_g = Rmat * n_temp;

        }
     
     if (u < cols && v < rows)
      {
        float z = vmap.ptr (v + 2 * rows)[u]; // load and convert: mm -> meters

        if (z != 0)
        {
          float rx = (1/rsqrtf(2)) * ((v_g.z/fx)/n_g.z);
          float ry = (1/rsqrtf(2)) * ((v_g.z/fy)/n_g.z);
          float rz = z;

          rmap.ptr (v                 )[u] = rx;
          rmap.ptr (v + rows    )[u] = ry;
          rmap.ptr (v + rows * 2)[u] = rz;
        }
        else
          rmap.ptr (v)[u] = numeric_limits<float>::quiet_NaN ();
	  //printf(" All going well in ComputeRmapKernel ");

      }
    }*/
	
  //}
//}
//----------------------------------------------------------------------------------------------------------------------------------
/*Copyright (C) 2016 Multimedia Communication Laboratory - Illinois Institute of Technology, Chicago*/

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void
    computeRegionKernel (int x, int y, const PtrStepSz<unsigned short> depth, PtrStep<unsigned char> dynmap, PtrStep<unsigned short> rgnmap)
	 {
		 
		 int u = threadIdx.x + blockIdx.x * blockDim.x;
         int v = threadIdx.y + blockIdx.y * blockDim.y;
		 
		 //printf("%d ", dynmap.ptr(y)[x]);
		 if ( u > depth.cols || u < 0 || v > depth.rows || v < 0)
			return;
		
		 if (dynmap.ptr(v)[u] != 255)
             rgnmap.ptr(v)[u] = 0;

		 else if (dynmap.ptr(v)[u] == 255)
		{
			//printf("inside first if");
			//down
			if ((abs(depth.ptr(v)[u] - depth.ptr(v+1)[u])<= 3))
				{
				
					dynmap.ptr(v+1)[u] = 255;
					rgnmap.ptr(v+1)[u] = depth.ptr(v+1)[u];
					//printf("in down ");
				}

			//right
			if ((abs(depth.ptr(v)[u] - depth.ptr(v)[u+1])<= 3)) 
				{
				
					dynmap.ptr(v)[u+1] = 255;
					rgnmap.ptr(v)[u+1] = depth.ptr(v)[u+1];
					//printf("in right ");
				
				}

			//up
			if ((abs(depth.ptr(v)[u] - depth.ptr(v-1)[u])<= 3)) 
				{
				
					dynmap.ptr(v-1)[u] = 255;
					rgnmap.ptr(v-1)[u] = depth.ptr(v-1)[u];
					//printf("in up ");
				
				}

			//left
			if ((abs(depth.ptr(v)[u] - depth.ptr(v)[u-1]) <= 3))
				{
				
					dynmap.ptr(v)[u-1] = 255;
					rgnmap.ptr(v)[u-1] = depth.ptr(v)[u-1];
					//printf("in left ");
				
				}
			//printf("in region growing kernel");
		}
	 }

  }
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void
	computeEroisonKernel (int rows, int cols, PtrStep<unsigned char> src) 
   {
	   int u = threadIdx.x + blockIdx.x * blockDim.x;
	   int v = threadIdx.y + blockIdx.y * blockDim.y;

	   if ( u >= cols - 1 || v >= rows - 1 || u <= 0 || v <= 0) // excluding first row/col and last row/col
		   return;

	   /*if (src.ptr(v)[u] == 0)
	      {
			  dst.ptr(v)[u] == 0;
	      }*/

	   if (src.ptr(v)[u] == 255 && src.ptr(v-1)[u] == 255 && src.ptr(v+1)[u] == 255 &&
		        src.ptr(v-1)[u-1] == 255 && src.ptr(v)[u-1] == 255 && src.ptr(v+1)[u-1] == 255 && 
		        src.ptr(v-1)[u+1] == 255 && src.ptr(v)[u+1] == 255 && src.ptr(v+1)[u+1] == 255)
	      {
			  src.ptr(v)[u] = 255;
	      }

	   else 
		   src.ptr(v)[u] = 0;
   }



///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void
    computeICPmapKernel (int rows, int cols, PtrStep<unsigned char> icpmap)
    {
      int u = threadIdx.x + blockIdx.x * blockDim.x;
      int v = threadIdx.y + blockIdx.y * blockDim.y;

      if (u >= cols || v >= rows)
        return;

      icpmap.ptr(v)[u] = 0;
    }

__global__ void
    computeRMapKernel (int rows, int cols, PtrStep<float> rmap)
    {
      int u = threadIdx.x + blockIdx.x * blockDim.x;
      int v = threadIdx.y + blockIdx.y * blockDim.y;

      if (u >= cols || v >= rows)
        return;

      rmap.ptr(v)[u] = 0.f;
	  //rmap.ptr(v + rows)[u] = 0.f;
	  //rmap.ptr(v + 2 * rows)[u] = 0.f;
    }

__global__ void
    computeCMapKernel (int rows, int cols, PtrStep<float> cmap)
    {
      int u = threadIdx.x + blockIdx.x * blockDim.x;
      int v = threadIdx.y + blockIdx.y * blockDim.y;

      if (u >= cols || v >= rows)
        return;

      cmap.ptr(v)[u] = 0.f;
	  //cmap.ptr(v + rows)[u] = 0.f;
	  //cmap.ptr(v + 2 * rows)[u] = 0.f;
    }

__global__ void
    computeDynMapKernel (int rows, int cols, PtrStep<unsigned char> dynmap)
    {
      int u = threadIdx.x + blockIdx.x * blockDim.x;
      int v = threadIdx.y + blockIdx.y * blockDim.y;

      if (u >= cols || v >= rows)
        return;

      dynmap.ptr(v)[u] = 0;
    }

//--------------------------------------------------------------------------------------------------------------------------------------------------------

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::device::createVMap (const Intr& intr, const DepthMap& depth, MapArr& vmap)
{
  vmap.create (depth.rows () * 3, depth.cols ());

  dim3 block (32, 8);
  dim3 grid (1, 1, 1);
  grid.x = divUp (depth.cols (), block.x);
  grid.y = divUp (depth.rows (), block.y);

  float fx = intr.fx, cx = intr.cx;
  float fy = intr.fy, cy = intr.cy;
  
  //printf(" All going well in createVmap ");

  computeVmapKernel<<<grid, block>>>(depth, vmap, 1.f / fx, 1.f / fy, cx, cy);
  cudaSafeCall (cudaGetLastError ());
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::device::createNMap (const MapArr& vmap, MapArr& nmap)
{
  nmap.create (vmap.rows (), vmap.cols ());

  int rows = vmap.rows () / 3;
  int cols = vmap.cols ();

  dim3 block (32, 8);
  dim3 grid (1, 1, 1);
  grid.x = divUp (cols, block.x);
  grid.y = divUp (rows, block.y);
  
  //printf(" All going well in createNmap ");

  computeNmapKernel<<<grid, block>>>(rows, cols, vmap, nmap);
  cudaSafeCall (cudaGetLastError ());
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------
/*Copyright (C) 2016 Multimedia Communications Laboratory - Illinoins Institute of Technology, Chicago*/

// modification for radius map computation
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::device::createRMap(const DepthMap& depth, MapArr& rmap)
{
  rmap.create (depth.rows () * 3, depth.cols ());

  /*int rows = depth.rows() * 3;
  int cols = depth.cols ();

  dim3 block (32, 8);
  dim3 grid (1, 1, 1);
  grid.x = divUp (cols, block.x);
  grid.y = divUp (rows, block.y);

  computeRMapKernel<<<grid, block>>>(rows, cols, rmap);
  cudaSafeCall (cudaGetLastError ());*/
}

void
pcl::device::createIcpMap(const DepthMap& depth, ICPStatusMap& icpmap)
{
  icpmap.create (depth.rows () , depth.cols ());

  int rows = depth.rows();
  int cols = depth.cols ();

  dim3 block (32, 8);
  dim3 grid (1, 1, 1);
  grid.x = divUp (cols, block.x);
  grid.y = divUp (rows, block.y);

  computeICPmapKernel<<<grid, block>>>(rows, cols, icpmap);
  cudaSafeCall (cudaGetLastError ());
}
  

void
pcl::device::createDynMap(const DepthMap& depth, DynamicMap& dynmap)
{
  dynmap.create (depth.rows () , depth.cols ());

  /*int rows = depth.rows();
  int cols = depth.cols ();

  dim3 block (32, 8);
  dim3 grid (1, 1, 1);
  grid.x = divUp (cols, block.x);
  grid.y = divUp (rows, block.y);

  computeDynMapKernel<<<grid, block>>>(rows, cols, dynmap);
  cudaSafeCall (cudaGetLastError ());*/
}

void
pcl::device::createRgnMap(const DepthMap& depth, DepthMap& rgnmap)
{
  rgnmap.create (depth.rows () , depth.cols ());

  /*int rows = depth.rows();
  int cols = depth.cols ();

  dim3 block (32, 8);
  dim3 grid (1, 1, 1);
  grid.x = divUp (cols, block.x);
  grid.y = divUp (rows, block.y);

  computeDynMapKernel<<<grid, block>>>(rows, cols, dynmap);
  cudaSafeCall (cudaGetLastError ());*/
}
  

void
pcl::device::createCMap(const DepthMap& depth, MapArr& cmap)
{
	cmap.create(depth.rows () * 3, depth.cols ());

	/*int rows = depth.rows() * 3;
    int cols = depth.cols ();

    dim3 block (32, 8);
    dim3 grid (1, 1, 1);
    grid.x = divUp (cols, block.x);
    grid.y = divUp (rows, block.y);

    computeCMapKernel<<<grid, block>>>(rows, cols, cmap);
    cudaSafeCall (cudaGetLastError ());*/
}

void
pcl::device::RegionGrowing(const DepthMap& depth, DynamicMap& dynmap, DepthMap& rgnmap)
{
	//cmap.create(depth.rows () * 3, depth.cols ());

	int rows = depth.rows();
    int cols = depth.cols ();

    dim3 block (32, 8);
    dim3 grid (1, 1, 1);
    grid.x = divUp (cols, block.x);
    grid.y = divUp (rows, block.y);
	// printf("in region");

    computeRegionKernel<<<grid, block>>>(rows, cols, depth, dynmap, rgnmap);
    cudaSafeCall (cudaGetLastError ());
}
 
void
pcl::device::erosion(const DynamicMap& depth_ero)
{
	int rows = depth_ero.rows();
	int cols = depth_ero.cols();

	dim3 block (32, 8);
	dim3 grid (1,1,1);
	grid.x = divUp (cols, block.x);
	grid.y = divUp (rows, block.y);

	computeEroisonKernel<<<grid, block>>>(rows, cols, depth_ero);
    cudaSafeCall (cudaGetLastError());
}

//-------------------------------------------------------------------------------------------------------------------------------------------

namespace pcl
{
  namespace device
  {
    __global__ void
    tranformMapsKernel (int rows, int cols, const PtrStep<float> vmap_src, const PtrStep<float> nmap_src,
                        const Mat33 Rmat, const float3 tvec, PtrStepSz<float> vmap_dst, PtrStep<float> nmap_dst)
    {
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;

      const float qnan = pcl::device::numeric_limits<float>::quiet_NaN ();

      if (x < cols && y < rows)
      {
        //vetexes
        float3 vsrc, vdst = make_float3 (qnan, qnan, qnan);
        vsrc.x = vmap_src.ptr (y)[x];

        if (!isnan (vsrc.x))
        {
          vsrc.y = vmap_src.ptr (y + rows)[x];
          vsrc.z = vmap_src.ptr (y + 2 * rows)[x];

          vdst = Rmat * vsrc + tvec;

          vmap_dst.ptr (y + rows)[x] = vdst.y;
          vmap_dst.ptr (y + 2 * rows)[x] = vdst.z;
        }

        vmap_dst.ptr (y)[x] = vdst.x;

        //normals
        float3 nsrc, ndst = make_float3 (qnan, qnan, qnan);
        nsrc.x = nmap_src.ptr (y)[x];

        if (!isnan (nsrc.x))
        {
          nsrc.y = nmap_src.ptr (y + rows)[x];
          nsrc.z = nmap_src.ptr (y + 2 * rows)[x];

          ndst = Rmat * nsrc;

          nmap_dst.ptr (y + rows)[x] = ndst.y;
          nmap_dst.ptr (y + 2 * rows)[x] = ndst.z;
        }

        nmap_dst.ptr (y)[x] = ndst.x;
      }
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::device::tranformMaps (const MapArr& vmap_src, const MapArr& nmap_src,
                           const Mat33& Rmat, const float3& tvec,
                           MapArr& vmap_dst, MapArr& nmap_dst)
{
  int cols = vmap_src.cols ();
  int rows = vmap_src.rows () / 3;

  vmap_dst.create (rows * 3, cols);
  nmap_dst.create (rows * 3, cols);

  dim3 block (32, 8);
  dim3 grid (1, 1, 1);
  grid.x = divUp (cols, block.x);
  grid.y = divUp (rows, block.y);

  tranformMapsKernel<<<grid, block>>>(rows, cols, vmap_src, nmap_src, Rmat, tvec, vmap_dst, nmap_dst);
  cudaSafeCall (cudaGetLastError ());
  //printf(" All going well in transformMaps ");

  cudaSafeCall (cudaDeviceSynchronize ());
}

namespace pcl
{
  namespace device
  {
    template<bool normalize>
    __global__ void
    resizeMapKernel (int drows, int dcols, int srows, const PtrStep<float> input, PtrStep<float> output)
    {
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;

      if (x >= dcols || y >= drows)
        return;

      const float qnan = numeric_limits<float>::quiet_NaN ();

      int xs = x * 2;
      int ys = y * 2;

      float x00 = input.ptr (ys + 0)[xs + 0];
      float x01 = input.ptr (ys + 0)[xs + 1];
      float x10 = input.ptr (ys + 1)[xs + 0];
      float x11 = input.ptr (ys + 1)[xs + 1];

      if (isnan (x00) || isnan (x01) || isnan (x10) || isnan (x11))
      {
        output.ptr (y)[x] = qnan;
        return;
      }
      else
      {
        float3 n;

        n.x = (x00 + x01 + x10 + x11) / 4;

        float y00 = input.ptr (ys + srows + 0)[xs + 0];
        float y01 = input.ptr (ys + srows + 0)[xs + 1];
        float y10 = input.ptr (ys + srows + 1)[xs + 0];
        float y11 = input.ptr (ys + srows + 1)[xs + 1];

        n.y = (y00 + y01 + y10 + y11) / 4;

        float z00 = input.ptr (ys + 2 * srows + 0)[xs + 0];
        float z01 = input.ptr (ys + 2 * srows + 0)[xs + 1];
        float z10 = input.ptr (ys + 2 * srows + 1)[xs + 0];
        float z11 = input.ptr (ys + 2 * srows + 1)[xs + 1];

        n.z = (z00 + z01 + z10 + z11) / 4;

        if (normalize)
          n = normalized (n);

        output.ptr (y        )[x] = n.x;
        output.ptr (y + drows)[x] = n.y;
        output.ptr (y + 2 * drows)[x] = n.z;
      }
    }

    template<bool normalize>
    void
    resizeMap (const MapArr& input, MapArr& output)
    {
      int in_cols = input.cols ();
      int in_rows = input.rows () / 3;

      int out_cols = in_cols / 2;
      int out_rows = in_rows / 2;

      output.create (out_rows * 3, out_cols);

      dim3 block (32, 8);
      dim3 grid (divUp (out_cols, block.x), divUp (out_rows, block.y));
      resizeMapKernel<normalize><< < grid, block>>>(out_rows, out_cols, in_rows, input, output);
      cudaSafeCall ( cudaGetLastError () );
      cudaSafeCall (cudaDeviceSynchronize ());
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::device::resizeVMap (const MapArr& input, MapArr& output)
{
  resizeMap<false>(input, output);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::device::resizeNMap (const MapArr& input, MapArr& output)
{
  resizeMap<true>(input, output);
}

namespace pcl
{
  namespace device
  {

    template<typename T>
    __global__ void
    convertMapKernel (int rows, int cols, const PtrStep<float> map, PtrStep<T> output)
    {
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;

      if (x >= cols || y >= rows)
        return;

      const float qnan = numeric_limits<float>::quiet_NaN ();

      T t;
      t.x = map.ptr (y)[x];
      if (!isnan (t.x))
      {
        t.y = map.ptr (y + rows)[x];
        t.z = map.ptr (y + 2 * rows)[x];
      }
      else
        t.y = t.z = qnan;

      output.ptr (y)[x] = t;
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T> void
pcl::device::convert (const MapArr& vmap, DeviceArray2D<T>& output)
{
  int cols = vmap.cols ();
  int rows = vmap.rows () / 3;

  output.create (rows, cols);

  dim3 block (32, 8);
  dim3 grid (divUp (cols, block.x), divUp (rows, block.y));

  convertMapKernel<T><< < grid, block>>>(rows, cols, vmap, output);
  cudaSafeCall ( cudaGetLastError () );
  cudaSafeCall (cudaDeviceSynchronize ());
}

template void pcl::device::convert (const MapArr& vmap, DeviceArray2D<float4>& output);
template void pcl::device::convert (const MapArr& vmap, DeviceArray2D<float8>& output);

namespace pcl
{
  namespace device
  {
    __global__ void
    mergePointNormalKernel (const float4* cloud, const float8* normals, PtrSz<float12> output)
    {
      int idx = threadIdx.x + blockIdx.x * blockDim.x;

      if (idx < output.size)
      {
        float4 p = cloud[idx];
        float8 n = normals[idx];

        float12 o;
        o.x = p.x;
        o.y = p.y;
        o.z = p.z;

        o.normal_x = n.x;
        o.normal_y = n.y;
        o.normal_z = n.z;

        output.data[idx] = o;
      }
    }
  }
}

void
pcl::device::mergePointNormal (const DeviceArray<float4>& cloud, const DeviceArray<float8>& normals, const DeviceArray<float12>& output)
{
  const int block = 256;
  int total = (int)output.size ();

  mergePointNormalKernel<<<divUp (total, block), block>>>(cloud, normals, output);
  cudaSafeCall ( cudaGetLastError () );
  cudaSafeCall (cudaDeviceSynchronize ());
}

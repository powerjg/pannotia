


#define BIGNUM 99999999

/**
* init kernel
* @param s_array   set array
* @param c_array   status array
* @param cu_array  status update array
* @param num_nodes number of vertices
* @param num_edges number of edges
*/
__kernel void init(__global int *s_array,
         	   __global int *c_array, 
                   __global int *cu_array, 
                            int num_nodes,
                            int num_edges){

    //get my workitem id
    int tid = get_global_id(0);
    if (tid < num_nodes){
        //set the status array: not processed
        c_array[tid]  = -1;
        cu_array[tid] = -1;
        s_array[tid] = 0;
    }
}
			
/**
* mis1 kernel
* @param row          csr pointer array
* @param col          csr column index array
* @param node_value   node value array
* @param s_array      set array
* @param c_array node status array
* @param min_array    node value array
* @param stop node    value array
* @param num_nodes    number of vertices
* @param num_edges    number of edges
*/
__kernel void mis1(  __global int *row, 
                     __global int *col, 
                     __global float *node_value,
                     __global int *s_array,
                     __global int *c_array,
                     __global float *min_array,
                     __global int *stop,
                              int num_nodes,
                              int num_edges){

    //get workitem id
    int tid = get_global_id(0);
    if (tid < num_nodes){
       //if the vertex is not processed
       if(c_array[tid] == -1){       
	    *stop = 1;
            //get the start and end pointers
	    int start = row[tid];
	    int end;
            if (tid + 1 < num_nodes)
               end = row[tid + 1] ;
            else
               end = num_edges;

            //navigate the neighbor list and find the min
	    float min = BIGNUM;
	    for(int edge = start; edge < end; edge++){
	        if (c_array[col[edge]] == -1){
                    if(node_value[col[edge]] < min)
                      min = node_value[col[edge]]; 
                }
            }
            min_array[tid] = min;
        }
    }
}

/**
* mis2 kernel
* @param row          csr pointer array
* @param col          csr column index array
* @param node_value   node value array
* @param s_array      set array
* @param c_array      status array
* @param cu_array     status update array
* @param min_array    node value array
* @param num_nodes    number of vertices
* @param num_edges    number of edges
*/
__kernel void  mis2(  __global int *row, 
                      __global int *col, 
                      __global float *node_value,	
                      __global int *s_array,
                      __global int *c_array,
                      __global int *cu_array,
                      __global float *min_array,
                               int num_nodes,
                               int num_edges)
{

    //get my workitem id
    int tid = get_global_id(0);
    if (tid < num_nodes){

       if(node_value[tid] < min_array[tid]  && c_array[tid] == -1){  
          // -1: not processed -2: inactive 2: independent set
          //put the item into the independent set   
          s_array[tid] = 2;

          //get the start and end pointers
          int start = row[tid];
          int end;

          if (tid + 1 < num_nodes)
             end = row[tid + 1] ;
          else
             end = num_edges;

          //set the status to inactive
          c_array[tid] = -2;

          //mark all the neighnors inactive
          for(int edge = start; edge < end; edge++){
             if (c_array[col[edge]] == -1)
                 //use status update array to avoid race
	         cu_array[col[edge]] = -2;
	  }

       }	
    }
}

/**
* mis3 kernel
* @param cu_array     status update array
* @param  c_array     status array
* @param num_nodes    number of vertices
*/
__kernel void  mis3(  __global int *cu_array, 
                      __global int *c_array, 
                               int num_nodes)
{

    //get my workitem id
    int tid = get_global_id(0);
    //set the status array
    if (tid < num_nodes && cu_array[tid] == -2)
        c_array[tid] = cu_array[tid];
}



/************************************************************************************\
 *                                                                                  *
 * Copyright © 2014 Advanced Micro Devices, Inc.                                    *
 * All rights reserved.                                                             *
 *                                                                                  *
 * Redistribution and use in source and binary forms, with or without               *
 * modification, are permitted provided that the following are met:                 *
 *                                                                                  *
 * You must reproduce the above copyright notice.                                   *
 *                                                                                  *
 * Neither the name of the copyright holder nor the names of its contributors       *   
 * may be used to endorse or promote products derived from this software            *
 * without specific, prior, written permission from at least the copyright holder.  *
 *                                                                                  *
 * You must include the following terms in your license and/or other materials      *
 * provided with the software.                                                      * 
 *                                                                                  *  
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"      *
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE        *
 * IMPLIED WARRANTIES OF MERCHANTABILITY, NON-INFRINGEMENT, AND FITNESS FOR A       *
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER        *
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,         *
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT  * 
 * OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS      *
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN          *
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING  *
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY   *
 * OF SUCH DAMAGE.                                                                  *
 *                                                                                  *
 * Without limiting the foregoing, the software may implement third party           *  
 * technologies for which you must obtain licenses from parties other than AMD.     *  
 * You agree that AMD has not obtained or conveyed to you, and that you shall       *
 * be responsible for obtaining the rights to use and/or distribute the applicable  * 
 * underlying intellectual property rights related to the third party technologies. *  
 * These third party technologies are not licensed hereunder.                       *
 *                                                                                  *
 * If you use the software (in whole or in part), you shall adhere to all           *        
 * applicable U.S., European, and other export laws, including but not limited to   *
 * the U.S. Export Administration Regulations ("EAR") (15 C.F.R Sections 730-774),  *
 * and E.U. Council Regulation (EC) No 428/2009 of 5 May 2009.  Further, pursuant   *
 * to Section 740.6 of the EAR, you hereby certify that, except pursuant to a       *
 * license granted by the United States Department of Commerce Bureau of Industry   *
 * and Security or as otherwise permitted pursuant to a License Exception under     *
 * the U.S. Export Administration Regulations ("EAR"), you will not (1) export,     *
 * re-export or release to a national of a country in Country Groups D:1, E:1 or    * 
 * E:2 any restricted technology, software, or source code you receive hereunder,   * 
 * or (2) export to Country Groups D:1, E:1 or E:2 the direct product of such       *
 * technology or software, if such foreign produced direct product is subject to    * 
 * national security controls as identified on the Commerce Control List (currently * 
 * found in Supplement 1 to Part 774 of EAR).  For the most current Country Group   * 
 * listings, or for additional information about the EAR or your obligations under  *
 * those regulations, please refer to the U.S. Bureau of Industry and Security's    *
 * website at http://www.bis.doc.gov/.                                              *
 *                                                                                  *
\************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <algorithm>
#include "../../graph_parser/parse.h"
#include "../../graph_parser/util.h"

#include "kernel/kernel.h"

#define RANGE 2048

void dump2file(int *adjmatrix, int num_nodes);
void print_vector(int *vector, int num);
void print_vectorf(float *vector, int num);

int main(int argc, char **argv){
	
    char *tmpchar;	

    int num_nodes; 
    int num_edges;
    int file_format = 1;
    bool directed = 0;

    //input arguments
    if(argc == 3){
       tmpchar =  argv[1];  //graph inputfile
       file_format = atoi(argv[2]); //choose file format
    }
    else{
       fprintf(stderr, "You did something wrong!\n");
       exit(1);
    }	
	
    srand(7);

    //allocate the csr array
    csr_array *csr = (csr_array *)malloc(sizeof(csr_array));
    if(!csr) fprintf(stderr, "malloc failed csr\n");

    //parse the graph into the csr structure
    if (file_format == 1)
       csr = parseMetis(tmpchar, &num_nodes, &num_edges, directed);
    else if (file_format == 0)
       csr = parseCOO(tmpchar, &num_nodes, &num_edges, directed);
    else{
       fprintf(stderr, "reserve for future");
       exit(1);
    }

    //allocate the node value array
    float *node_value = (float *)malloc(num_nodes * sizeof(float));
    if(!node_value) fprintf(stderr, "malloc failed node_value\n");

    //allocate the set array
    int *s_array    = (int *)malloc(num_nodes * sizeof(int));
    if(!s_array) fprintf(stderr, "malloc failed s_array\n");

    //randomize the node values
    for(int i = 0; i < num_nodes; i++)
       node_value[i] =  rand()/(float)RAND_MAX;
    
    // c array?
    int *c_array    = (int *)malloc(num_nodes * sizeof(int));
    if(!c_array) fprintf(stderr, "malloc failed c_array\n");

    // c update array?
    int *cu_array    = (int *)malloc(num_nodes * sizeof(int));
    if(!cu_array) fprintf(stderr, "malloc failed cu_array\n");
    
    // c update array?
    float *min_array    = (float *)malloc(num_nodes * sizeof(float));
    if(!min_array) fprintf(stderr, "malloc failed min_array\n");

    double time1 = gettime();

    //OpenCL dimensions
    int block_size = 128;
    int global_size = (num_nodes%block_size == 0)? num_nodes: (num_nodes/block_size + 1) * block_size;
    printf("global_size = %d\n", global_size);	

    SNK_INIT_LPARM(lparam, global_size);
    lparam->ldims[0] = block_size;
    init(s_array, c_array, cu_array, num_nodes, num_edges, lparam);

    //termination variable
    int stop = 1;
    while(stop){
        
        stop = 0;
        
        mis1(csr->row_array, csr->col_array, node_value, s_array, c_array, min_array, &stop, num_nodes, num_edges, lparam);
        mis2(csr->row_array, csr->col_array, node_value, s_array, c_array, cu_array, min_array, num_nodes, num_edges, lparam);
        mis3(cu_array, c_array, num_nodes, lparam);
    }

    double time2=gettime();
    //print out the timing characterisitics
    printf("kernel + memcpy time %f ms\n",  (time2-time1) * 1000);

#if 1 
    //print the set array
    print_vector(s_array, num_nodes);
#endif

    //clean up the host-side arrays
    free(node_value);
    free(s_array);
    free(c_array);
    free(cu_array);
    free(min_array);
    csr->freeArrays();
    free(csr);

    return 0;

}

void print_vector(int *vector, int num){

    FILE * fp = fopen("result.out", "w"); 
    if(!fp) { printf("ERROR: unable to open result.txt\n");}

    for(int i = 0; i < num; i++){
        fprintf(fp, "%d\n", vector[i]);
    }

    fclose(fp);

}

void print_vectorf(float *vector, int num){

    FILE * fp = fopen("result.out", "w"); 
    if(!fp) { printf("ERROR: unable to open result.txt\n");}

    for(int i = 0; i < num; i++){
        fprintf(fp, "%f\n", vector[i]);
    }

    fclose(fp);

}

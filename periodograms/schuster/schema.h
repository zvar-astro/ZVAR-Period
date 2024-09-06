/* 
Code for 
*/


#ifndef SCHEMA_BUILDER_H_
#define SCHEMA_BUILDER_H_
 
typedef struct starlist {
    unsigned long source_id;
    float ra;
    float dec;
} starlist;


typedef struct ptable {
    float period_LS;
    float height_LS;
    float significance_LS;
} ptable;


#endif

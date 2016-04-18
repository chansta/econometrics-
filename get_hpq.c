//#include<stdio.h>
#include<math.h> 

double dotprod(double a[], double b[], int r) {
int i;
double final;

final=0; 
for (i=0;i<r;++i) {
final += a[i]*b[i]; 
//printf("a is %f\n", b[i]); 
} 
return final; 
}

void get_hpq(h,w,a,b,e2,h0,order) 
double h[], w[], a[], b[], e2[], h0[];
int order[];
//int *pt, *qt, *per;
{
  int i,j, p, q, period, r; 
  double e2_temp[order[0]], h_temp[order[1]];
  double sr, lr;
  
  p = order[0];
  q = order[1];
  period = order[2]; 
  //printf("p is %i\n",p);
  for (i=0;i<q;++i) {h[i] = h0[i];}
  if (p>q) {r=p;} else {r=q;} 
  for (i=0;i<q;++i) {h[i] = h0[i];}
  if (p>q) {r=p;} else {r=q;} 
  for (i=0;i<r;++i) {
  h[i] = 0; 
  }
  for (i=r-q;i<r;++i) {
  h[i] = h0[i];
  }
  for (i=r;i<period;++i) {
    for(j=0;j<p;++j) {
    e2_temp[j] = e2[i-1-j];
    }
    for(j=0;j<q;++j) {
    h_temp[j] = h[i-1-j];
    }
   sr = dotprod(a,e2_temp,p);
   lr = dotprod(b,h_temp,q);
   h[i] = w[0] + sr + lr; 
  }
}
/*
main() {
  int i, k, p, q, period, r; 
  double w, a[1], b[1], e2[10], e[10], h[10], h0[1];
  double e2_temp[*pt], h_temp[*qt]; 
  
  p = 1;
  q = 1;
  w = 0.1; 
  a[0] = 0.2;
  b[0] = 0.75; 
  period = 10.0; 
  h0[0] = 0.1; 
  for (i=0;i<period;++i) {
  e[i] = -0.25*(1/10.0)*i; 
  e2[i] = e[i]*e[i]; 
  }
  if (p>q) {r=p;} else {r=q;} 
  for (i=0;i<q;++i) {h[i] = h0[i];}
  if (p>q) {r=p;} else {r=q;} 
  for (i=r;i<period;++i) {
    for (k=0;k<p;++k) {
    e2_temp[k] = e2[i-1-k];
    }
    for (k=0;k<q;++k) {
    h_temp[k] = h[i-1-k];
    }
   sr = dotprod(a,e2_temp,p);
   lr = dotprod(b,h_temp,q);
   h[i] = w + sr + lr; 
  printf("h[%d] is %f\n",i, h[i]); 
  } 
return 0; 
}
*/

/*  
	dp training： 
	4*4 方格：(1,1) 为初始状态- (4,4) 为终止状态
	除了到达终止状态，其余所有动作的收益均为 1
	求解最优策略 
*/

#include<iostream>
#include<map>
using namespace std;
const int stateN=16; // state number = 16
const double delta = 1.; 


map<pair<int, string>,double>pi; // s,a
double rewards[2]={-1,0};
string actions[4]={"up","down","left","right"};

double v[stateN+1]={0};
double v_[stateN+1]={0};
void define_pi(){
	for(int i=0;i<16;i++)
	{
		for(int j=0;j<4;j++)
		{
			pi[pair<int,string>(i,actions[j])]=0.25;
		}
	}
}
double define_pro(int k, int r, int s, int a){
	if(a==0){
		if( (s>3 && k==s-4 && r==-1) || (s<=3 && s==k && r==-1))
			return 1;
	}
	if(a==1){
//		if(s==11 && k==15 && r==0)
//			return 1;
		if( (s<12 && k==s+4 && r==-1) || (s>=12 && s==k && r==-1))
			return 1;
	}
	if(a==2){
		if( (s%4!=0 && k==s-1 && r==-1) || (s%4==0 && s==k && r==-1) )
			return 1;
	}
	if(a==3){
//		if(s==14 && k==15 && r==0)
//			return 1;
		if( ((s+1)%4!=0 && k==s+1 && r==-1) || ((s+1)%4==0 && s==k && r==-1) )
			return 1;
	}
	return 0;
}
int main(){
	define_pi();
	
for(int i=0;i<1000;i++){
	for(int s=0;s<stateN;s++) 
	{
		if(s==stateN-1 || s==0) continue;
		double sum=0.;
		for(int i=0;i<4;i++) //four action
		{
			double pi_=pi[{s,actions[i]}];
			double sum_=0.;
			for(int j=0;j<2;j++) //two rewards
			{
				double r=rewards[j];
				for(int k=0;k<stateN;k++) //k state
				{
					sum_+=define_pro(k,r,s,i)*(r+v[k]);
				}	 
			}
			sum+=pi_*sum_;	
		}
		v_[s]=sum;
	}
	for(int i=0;i<stateN;i++)
		v[i]=v_[i];
}
	
	for(int i=0;i<16;i++)
	{
		cout<<v[i]<<" ";
		if( (i+1)%4 == 0 ) cout<<"\n";
	}
	
	return 0;
} 

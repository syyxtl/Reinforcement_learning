/*
	摆渡人，摆渡共有1-7个横向位置
	到达点1获得reward=5，到达点7获得reward=10 
	求最佳策略 
*/ 

#include<iostream>
#include<map>
using namespace std;
const int stateN=7; // state number = 7

double rewards[3]={5,10,0};
string actions[]={"left", "right"};
map<pair<int, string>,double>pi;
double v[stateN]={0}; 


void define_pi(){ //pi(s,a)
	for(int i=0;i<stateN;i++) 
	{
		for(int j=0;j<2;j++) // 2 action
		{
			pi[pair<int,string>(i,actions[j])]=1;
		}
	}
}
double define_pro(int k, int r, int s, int a){ //s',r | s,a
	 if(k==1 && r==5 && s==0 && a==1) return 1;
	 if(k==5 && r==10 && s==6 && a==0) return 1;
	 return 0; 
}
int main(){
	define_pi();
	
	for(int num=0;num<1000;num++){
		for(int s=0;s<stateN;s++) 
		{
			double sum=0.0;
			for(int i=0;i<2;i++) // 2 action
			{
				double pi_=pi[{s,actions[i]}];
				double sum_=0.;
				for(int j=0;j<3;j++) //3 rewards
				{
					double r=rewards[j];
					for(int k=0;k<stateN;k++) //k state
					{
						sum_+=define_pro(k,r,s,i)*(r+v[k]);
					}	 
				}
				sum+=pi_*sum_;	
			}
			v[s]=sum;
		}
//	for(int i=0;i<stateN;i++)
//		v[i]=v_[i];
	}
	
	for(int i=0;i<7;i++)
	{
		cout<<v[i]<<" ";
	}cout<<"\n"; 
	
	return 0;
} 

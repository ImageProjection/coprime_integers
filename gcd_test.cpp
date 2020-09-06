#include <cstdio>

int main()
{
    int a,b,gcd;

    scanf("%d %d",&a,&b);
    while ((a!=0) && (b!=0))
    {
        if (a>b)
        {
            a%=b;
        }
        else
        {
            b%=a;
        }
    }
    if (a==0)
    {
        gcd=b;
    }
    else
    {
        gcd=a;
    }
    
    
    printf("gcd=%d\n",gcd);    
}
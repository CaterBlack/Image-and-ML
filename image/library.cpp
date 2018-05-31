#include <opencv2/opencv.hpp>
#include<iostream>
#include<algorithm>

using namespace std;
using namespace cv;

//灰度化
Mat gray(Mat img){
    Mat changeImg(img.rows,img.cols,CV_8U);

    for(int i=0;i<img.rows;i++){
        for(int j=0;j<img.cols;j++){
            uchar b=img.at<Vec3b>(i,j)[0];
            uchar g=img.at<Vec3b>(i,j)[1];
            uchar r=img.at<Vec3b>(i,j)[2];
            double gray= 0.3*r+0.59*g+0.11*b;
            changeImg.at<uchar>(i,j)=gray;

        }
    }
    return changeImg;
}

//soble滤波
Mat soble(Mat img){

    Mat changeImg(img.rows,img.cols,CV_8UC3);

    for(int i=0;i<img.rows;i++){
        for(int j=0;j<img.cols;j++){


            int filterX[3][3]={{-1,0,1},{-2,0,2},{-1,0,1}};
            int filterY[3][3]={{-1,-2,-1},{0,0,0},{1,2,1}};int x=i;
            int y=j;
            if(x<=0){
                x=1;
            }
            if(x>=img.rows-1){
                x=img.rows-2;
            }
            if(y<=0){
                y=1;
            }
            if(y>=img.cols-1){
                y=img.cols-2;
            }

            double fx=0.0,fy=0.0;
            for(int m=-1;m<2;m++){
                for(int n=-1;n<2;n++){
                    fx+=filterX[m+1][n+1]*img.at<Vec3b>(x+m,y+n)[0];
                    fy+=filterY[m+1][n+1]*img.at<Vec3b>(x+m,y+n)[0];
                    fx+=filterX[m+1][n+1]*img.at<Vec3b>(x+m,y+n)[1];
                    fy+=filterY[m+1][n+1]*img.at<Vec3b>(x+m,y+n)[1];
                    fx+=filterX[m+1][n+1]*img.at<Vec3b>(x+m,y+n)[2];
                    fy+=filterY[m+1][n+1]*img.at<Vec3b>(x+m,y+n)[2];
                }
            }
            fx=fx/3.0;
            fy=fy/3.0;
            double f=fabs(fx)+fabs(fy);

            Vec3b vecf={(uchar)f,(uchar)f,(uchar)f};
            changeImg.at<Vec3b>(i,j)=vecf;

        }
    }
    return changeImg;

}

//高斯滤波（）
Mat gaussianSmoothGray(Mat img){
    Mat changeImg(img.rows,img.cols,CV_8U);

    for(int i=0;i<img.rows;i++){
        for(int j=0;j<img.cols;j++){
            int x=i;
            int y=j;
            if(x<=0){
                x=1;
            }
            if(x>=img.rows-1){
                x=img.rows-2;
            }
            if(y<=0){
                y=1;
            }
            if(y>=img.cols-1){
                y=img.cols-2;
            }

            double filter[3][3]={{1/16.0,2/16.0,1/16.0},{2/16.0,4/16.0,2/16.0},{1/16.0,2/16.0,1/16.0}};

            double b=0;
            for(int m=-1;m<2;m++){
                for(int n=-1;n<2;n++){
                    b+=img.at<uchar>(x+m,y+n)*filter[m+1][n+1];
                }
            }

            changeImg.at<uchar>(i,j)=b;

        }
    }
    return changeImg;
}

//canny算子
Mat canny(Mat img){
    Mat gx(img.rows,img.cols,CV_64F);
    Mat gy(img.rows,img.cols,CV_64F);
    Mat Mxy(img.rows,img.cols,CV_64F);
    Mat gx2;
    Mat gy2;
    Mat Axy(img.rows,img.cols,CV_64F);
    //灰度化
    Mat grayImg=gray(img);

    //高斯平滑
    Mat gaussImg=gaussianSmoothGray(grayImg);

    //求gx gy
    for(int i=0;i<img.rows;i++){
        for(int j=0;j<img.cols;j++){
            int x=i;
            int y=j;
            if(x<=0){
                x=1;
            }
            if(x>=img.rows-1){
                x=img.rows-2;
            }
            if(y<=0){
                y=1;
            }
            if(y>=img.cols-1){
                y=img.cols-2;
            }

            int filterX[3][3]={{-1,0,1},{-2,0,2},{-1,0,1}};
            int filterY[3][3]={{-1,-2,-1},{0,0,0},{1,2,1}};


            double fx=0.0,fy=0.0;
            for(int m=-1;m<2;m++){
                for(int n=-1;n<2;n++){
                    fx+=filterX[m+1][n+1]*grayImg.at<uchar>(x+m,y+n);
                    fy+=filterY[m+1][n+1]*grayImg.at<uchar>(x+m,y+n);
                }
            }

            //            int filterX[2][2]={{-1,1},{-1,1}};
            //            int filterY[2][2]={{1,1},{-1,-1}};
            //            double fx=0,fy=0;
            //            for(int m=0;m<2;m++){
            //                for(int n=0;n<2;n++){
            //                    fx+=filterX[m][n]*grayImg.at<uchar>(x+m,y+n);
            //                    fy+=filterY[m][n]*grayImg.at<uchar>(x+m,y+n);
            //                }
            //            }

            gx.at<double>(i,j)=fx;
            gy.at<double>(i,j)=fy;
        }

    }

    //求梯度向量大小与方向

    cv::pow(gx,2,gx2);
    cv::pow(gy,2,gy2);
    cv::sqrt(gx2+gy2,Mxy);

    for(int i=0;i<img.rows;i++){
        for(int j=0;j<img.cols;j++){
            Axy.at<double>(i,j)=atan2(gx.at<double>(i,j),gy.at<double>(i,j))*180/3.1415926;
        }
    }

    //非最大值抑制
    Mat Mxy2(img.rows,img.cols,CV_64F);
    for(int i=0;i<img.rows;i++){
        for(int j=0;j<img.cols;j++){
            int x=i;
            int y=j;
            if(x<=0){
                x=1;
            }
            if(x>=img.rows-1){
                x=img.rows-2;
            }
            if(y<=0){
                y=1;
            }
            if(y>=img.cols-1){
                y=img.cols-2;
            }

            double angle=Axy.at<double>(x,y);
            double dx,dy;
            if(angle<0){
                angle=angle+180;
            }
            if(angle<=22.5||angle>=157.5){
                dx=1;
                dy=0;
            }else if(angle>=22.5&&angle<=67.5){
                dx=1;
                dy=1;
            }else if(angle>=67.5&&angle<=112.5){
                dx=0;
                dy=1;
            }else{
                dx=-1;
                dy=1;
            }

            double M=Mxy.at<double>(x,y);
            double ML=Mxy.at<double>(x+dx,y+dy);
            double MR=Mxy.at<double>(x-dx,y-dy);
            if(M>ML&&M>MR){
                Mxy2.at<double>(i,j)=M;
            }

        }
    }

    double max=0;
    for(int i=0;i<img.rows;i++){
        for(int j=0;j<img.cols;j++){
            if(Mxy2.at<double>(i,j)>max){
                max=Mxy2.at<double>(i,j);
            }
        }
    }
    cout<<max<<endl;

    //    for(int i=0;i<img.rows;i++){
    //        for(int j=0;j<img.cols;j++){
    //            cout<<Mxy2.at<double>(i,j)<<" ";
    //        }
    //    }

    //双阈值(普通法 不好)
    //    Mat gNH(img.rows,img.cols,CV_64F);
    //    Mat gNL(img.rows,img.cols,CV_64F);
    //    int nHist[1024];
    //    int nEdgeNum;
    //    int nMaxMag=0;
    //    for(int i=0;i<1024;i++){
    //        nHist[i]=0;
    //    }
    //    for(int i=0;i<img.rows;i++){
    //        for(int j=0;j<img.cols;j++){
    //            int ss=(int)Mxy2.at<double>(i,j);
    //            if(ss<1024){
    //                nHist[ss]++;
    //            }
    //        }
    //    }
    //    nEdgeNum=0;
    //    for(int i=1;i<1024;i++){
    //        if(nHist[i]!=0){
    //            nMaxMag=i;
    //        }
    //        nEdgeNum+=nHist[i];
    //    }

    //    int nThrHigh;
    //    int nThrLow;
    //    double dRateHigh=0.7;
    //    double dRateLow=0.5;
    //    int nHightCount=(int)(dRateHigh*nEdgeNum+0.5);
    //    int count=1;
    //    nEdgeNum=nHist[1];
    //    while((nEdgeNum<=nHightCount)&&(count<nMaxMag-1))
    //    {
    //        count++;
    //        nEdgeNum+=nHist[count];
    //    }
    //    nThrHigh=count;

    //    count=1;
    //    int nLowCount=(int)(nEdgeNum*dRateLow+0.5);
    //    nEdgeNum=nHist[1];
    //    while((nEdgeNum<=nLowCount)&&(count<nMaxMag-1))
    //    {
    //        count++;
    //        nEdgeNum+=nHist[count];
    //    }
    //    nThrLow=count;
    //    cout<<nThrHigh<<endl;
    //    cout<<nThrLow<<endl;

    //    for(int i=0;i<img.rows;i++){
    //        for(int j=0;j<img.cols;j++){
    //            if(Mxy2.at<double>(i,j)>nThrHigh){
    //                gNH.at<double>(i,j)=255;
    //            }
    //            if(Mxy2.at<double>(i,j)>nThrLow){
    //                gNL.at<double>(i,j)=255;
    //            }
    //        }
    //    }

    //otsu法求阈值
    //    Mat gNH(img.rows,img.cols,CV_64F);
    //    Mat gNL(img.rows,img.cols,CV_64F);
    //    int nHist[256];
    //    for(int i=0;i<256;i++){
    //        nHist[i]=0;
    //    }
    //    for(int i=0;i<img.rows;i++){
    //        for(int j=0;j<img.cols;j++){
    //            int ss=(int)Mxy2.at<double>(i,j);
    //            if(ss<256){
    //                nHist[ss]++;
    //            }
    //        }
    //    }
    //    double pHist[256]{0};
    //    for(int i=0;i<256;i++){
    //        pHist[i]=nHist[i]/(double(img.rows*img.cols));
    //    }
    //    double PHist[256]{0};
    //    for(int i=0;i<256;i++){
    //        for(int j=0;j<=i;j++){
    //            PHist[i]+=pHist[j];
    //        }
    //    }
    //    int MHist[256]{0};
    //    for(int i=0;i<256;i++){
    //        double ss=0;
    //        for(int j=0;j<=i;j++){
    //            ss+=(j*pHist[j]+0.5);
    //        }
    //        MHist[i]=int(ss);
    //    }
    //    int MG=MHist[255];
    //    long theta[256]{0};
    //    for(int i=0;i<256;i++){
    //        if((1-PHist[i])<=0){
    //            theta[i]=0;
    //        }else{
    //            double ss=pow((MG*PHist[i]-MHist[i]),2)/(PHist[i]*(1-PHist[i]));
    //            theta[i]=long(ss);
    //        }
    //    }

    //    int thre=0;
    //    max=0;
    //    for(int i=0;i<256;i++){
    //        if(theta[i]>max){
    //            max=theta[i];
    //            thre=i;
    //        }
    //    }
    //    cout<<thre<<endl;
    //    //    for(int i=0;i<256;i++){
    //    //        cout<<nHist[i]<<" ";
    //    //        if(i%20==0){
    //    //            cout<<endl;
    //    //        }
    //    //    }

    //自创法(形态学去噪)
    Mat gNH(img.rows,img.cols,CV_8U);
    for(int i=0;i<img.rows;i++){
        for(int j=0;j<img.cols;j++){
            if(Mxy2.at<double>(i,j)>255*0.25){
                gNH.at<uchar>(i,j)=255;
            }
        }
    }
    imshow("img",gNH);

    for(int i=0;i<img.rows;i++){
        for(int j=0;j<img.cols;j++){
            int x=i;
            int y=j;
            if(x<=1){
                x=2;
            }
            if(x>=img.rows-2){
                x=img.rows-3;
            }
            if(y<=1){
                y=2;
            }
            if(y>=img.cols-2){
                y=img.cols-3;
            }

            int k=0;
            for(int m=-2;m<3;m++){
                for(int n=-2;n<3;n++){
                    if(gNH.at<uchar>(x+m,y+n)==255){
                        k++;
                    }
                }
            }
            if(k<3){
                gNH.at<uchar>(x,y)=0;
            }
        }
    }



    imshow("img2",gNH);



    //        Mat Gxy(img.rows,img.cols,CV_8U);
    //        Mat Gxy2(img.rows,img.cols,CV_8U);
    //        for(int i=0;i<img.rows;i++){
    //            for(int j=0;j<img.cols;j++){
    //                Gxy.at<uchar>(i,j)=(uchar)Mxy.at<double>(i,j);
    //                Gxy2.at<uchar>(i,j)=(uchar)Mxy2.at<double>(i,j);
    //            }
    //        }
    //        imshow("img3",Gxy);
    //        imshow("img4",Gxy2);


}

//得到直方图
double** getHist(Mat img){
    double** hist=new double*[3];
    for(int i=0;i<3;i++){
        hist[i]=new double[256];
    }
    for(int i=0;i<3;i++){
        for(int j=0;j<256;j++){
            hist[i][j]=0;
        }
    }
    for(int i=0;i<img.rows;i++){
        for(int j=0;j<img.cols;j++){
            hist[0][img.at<Vec3b>(i,j)[0]]++;
            hist[1][img.at<Vec3b>(i,j)[1]]++;
            hist[2][img.at<Vec3b>(i,j)[2]]++;
        }
    }

    return hist;

}

//直方图归一化
double** normalizeHist(double** hist){
    int max=0;
    for(int i=0;i<3;i++){
        for(int j=0;j<256;j++){
            if(hist[i][j]>max){
                max=hist[i][j];
            }
        }
    }
    double ** hist2=hist;
    for(int i=0;i<3;i++){
        for(int j=0;j<256;j++){
            hist2[i][j]/=max;
        }
    }
    return hist2;
}


//直方图均衡
Mat histBalanced(Mat img){
    int MN=img.cols*img.rows;
    Mat changeImg(img.rows,img.cols,CV_8UC3);
    double **hist=getHist(img); //得到直方图
    double map[3][256]; //像素均衡映射数组

    for(int i=0;i<3;i++){
        for(int j=0;j<256;j++){
            map[i][j]=0;
        }
    }

    for(int i=0;i<3;i++){
        for(int j=0;j<256;j++){
            for(int k=0;k<=j;k++){
                map[i][j]+=hist[i][k];
            }
            map[i][j]=map[i][j]*255/MN;

            if(map[i][j]<0){
                map[i][j]=0;
            }else if(map[i][j]>255){
                map[i][j]=255;
            }
        }
    }

    for(int i=0;i<changeImg.rows;i++){
        for(int j=0;j<changeImg.cols;j++){
            changeImg.at<Vec3b>(i,j)[0]=(uchar)map[0][img.at<Vec3b>(i,j)[0]];
            changeImg.at<Vec3b>(i,j)[1]=(uchar)map[1][img.at<Vec3b>(i,j)[1]];
            changeImg.at<Vec3b>(i,j)[2]=(uchar)map[2][img.at<Vec3b>(i,j)[2]];
        }
    }
    return changeImg;
}

//直方图匹配
Mat getHistRegulation(Mat img,Mat img2){
    int MN=img.cols*img.rows;
    Mat changeImg(img.rows,img.cols,CV_8UC3);
    Mat tempMat(img.rows,img.cols,CV_64FC3);
    double **hist1=getHist(img);
    double **hist2=getHist(img2);
    double map1[3][256];
    double map2[3][256];
    double map3[3][256];

    for(int i=0;i<3;i++){
        for(int j=0;j<256;j++){
            map1[i][j]=0;
            map2[i][j]=0;
            map3[i][j]=-1;
        }
    }

    for(int i=0;i<3;i++){
        for(int j=0;j<256;j++){
            for(int k=0;k<=j;k++){
                map1[i][j]+=hist1[i][k];
                map2[i][j]+=hist2[i][k];
            }
            map1[i][j]=map1[i][j]*255/MN;
            map2[i][j]=map2[i][j]*255/MN;

            if(map1[i][j]<0){
                map1[i][j]=0;
            }else if(map1[i][j]>255){
                map1[i][j]=255;
            }

            if(map2[i][j]<0){
                map2[i][j]=0;
            }else if(map2[i][j]>255){
                map2[i][j]=255;
            }
        }
    }

    for(int i=0;i<img.rows;i++){
        for(int j=0;j<img.cols;j++){
            tempMat.at<Vec3d>(i,j)[0]=map1[0][img.at<Vec3b>(i,j)[0]];
            tempMat.at<Vec3d>(i,j)[1]=map1[1][img.at<Vec3b>(i,j)[1]];
            tempMat.at<Vec3d>(i,j)[2]=map1[2][img.at<Vec3b>(i,j)[2]];
        }
    }


    for(int i=0;i<3;i++){
        for(int j=0;j<256;j++){
            map3[0][(int)map2[0][j]]=j;
            map3[1][(int)map2[1][j]]=j;
            map3[2][(int)map2[2][j]]=j;
        }
    }

    for(int i=0;i<3;i++){
        for(int j=0;j<256;j++){
            if(map3[i][j]==-1){
                int k=0;
                bool t=false;
                while(map3[i][j+k]==-1){
                    if(j+k>256){
                        k=-1;
                        t=true;
                    }
                    if(t){
                        k--;
                    }else{
                        k++;
                    }
                }
                map3[i][j]=map3[i][(j+k)%256];
            }
        }
    }


    for(int i=0;i<img.rows;i++){
        for(int j=0;j<img.cols;j++){
            changeImg.at<Vec3b>(i,j)[0]=(uchar)map3[0][(int)tempMat.at<Vec3d>(i,j)[0]];
            changeImg.at<Vec3b>(i,j)[1]=(uchar)map3[1][(int)tempMat.at<Vec3d>(i,j)[1]];
            changeImg.at<Vec3b>(i,j)[2]=(uchar)map3[2][(int)tempMat.at<Vec3d>(i,j)[2]];
        }
    }


    return changeImg;

}

/***********************************************************************/
//k均值
//点之间的距离
double getSim(Vec3b a,Vec3b b){
    double sum=0;
    for(int i=0;i<3;i++){
        sum+=(a[i]-b[i])*(a[i]-b[i]);
    }
    return sqrt(sum);
}

//生成随机聚类点
Vec3b* randCent(Mat img,int k){
    srand((unsigned)time(NULL));
    Vec3b* centroids=new Vec3b(k);
    for(int i=0;i<3;i++){
        int min=255,max=0;
        for(int r=0;r<img.rows;r++){
            for(int c=0;c<img.cols;c++){
                uchar cc=img.at<Vec3b>(r,c)[i];
                if(cc<min){
                    min=cc;
                }
                if(cc>max){
                    max=cc;
                }
            }
        }

        for(int j=0;j<k;j++){
            centroids[j][i]=rand()%(max-min)+min;
        }
    }
    return centroids;
}

//kmeans算法
Mat KMeans(Mat img,int k){
    int size=img.rows*img.cols;
    int*  clusterLabel=new int[size];
    Vec3b* centroids=randCent(img,k);
    bool change=true;
    int counter=0;



    while(counter<=50){
        counter++;
        cout<<"1"<<endl;
        change=false;
        int kkk=0;
        for(int r=0;r<img.rows;r++){
            for(int c=0;c<img.cols;c++){
                double minDis=10000000;
                int minIndex=-1;
                Vec3b ccc=img.at<Vec3b>(r,c);
                for(int j=0;j<k;j++){
                    double distJI=getSim(centroids[j],ccc);
                    if(distJI<minDis){
                        minDis=distJI;
                        minIndex=j;
                    }
                    if(clusterLabel[kkk]!=minIndex){
                        change=true;
                    }
                    clusterLabel[kkk]=minIndex;
                }

                kkk++;
            }
        }
        cout<<2<<endl;
        if(change){

            for(int i=0;i<k;i++){
                kkk=0;
                int num=1;
                int sum1=0,sum2=0,sum3=0;
                for(int r=0;r<img.rows;r++){
                    for(int c=0;c<img.cols;c++){
                        if(clusterLabel[kkk]==i){
                            Vec3b ccc=img.at<Vec3b>(r,c);
                            sum1+=ccc[0];
                            sum2+=ccc[1];
                            sum3+=ccc[2];
                            num++;
                        }

                        kkk++;
                    }
                }
                sum1=sum1/num;
                sum2=sum2/num;
                sum3=sum3/num;
                centroids[i]={sum1,sum2,sum3};
                cout<<centroids[i]<<endl;
            }
        }else{
            cout<<"GGGG"<<endl;
            break;
        }
    }

    int step=255/k;
    Mat changeImg(img.rows,img.cols,CV_8UC3);
    int kkk=0;
    cout<<3<<endl;
    Vec3b* color=new Vec3b[k];
    for(int i=0;i<k;i++){
        color[i]={uchar(step*i*(rand()%2)),uchar(step*i*(rand()%2)),uchar(step*i*(rand()%2))};
    }
    for(int i=0;i<img.rows;i++){
        for(int j=0;j<img.cols;j++){
            int type=int(clusterLabel[kkk]);
            Vec3b ccc=color[type];
            changeImg.at<Vec3b>(i,j)=ccc;
            kkk++;
        }
    }

    return changeImg;
}

/***********************************************************************/

//拉普拉斯滤波
Mat LaplaceFilter(Mat img){
    Mat changeImg(img.rows,img.cols,CV_8UC3);

    for(int i=0;i<img.rows;i++){
        for(int j=0;j<img.cols;j++){
            int x=i;
            int y=j;
            if(x<=0){
                x=1;
            }
            if(x>=img.rows-1){
                x=img.rows-2;
            }
            if(y<=0){
                y=1;
            }
            if(y>=img.cols-1){
                y=img.cols-2;
            }

            double filter[3][3]={{0,1,0},{1,-4,1},{0,1,0}};

            double b=0,g=0,r=0;
            for(int m=-1;m<2;m++){
                for(int n=-1;n<2;n++){
                    b+=img.at<Vec3b>(x+m,y+n)[0]*filter[m+1][n+1];
                    g+=img.at<Vec3b>(x+m,y+n)[1]*filter[m+1][n+1];
                    r+=img.at<Vec3b>(x+m,y+n)[2]*filter[m+1][n+1];
                }
            }
            b=img.at<Vec3b>(x,y)[0]+b;
            g=img.at<Vec3b>(x,y)[1]+g;
            r=img.at<Vec3b>(x,y)[2]+r;
            if(b<0){
                b=0;
            }
            if(b>255){
                b=255;
            }
            if(g<0){
                g=0;
            }
            if(g>255){
                g=255;
            }
            if(r<0){
                r=0;
            }
            if(r>255){
                r=255;
            }

            Vec3b vecf={(uchar)b,(uchar)g,(uchar)r};
            changeImg.at<Vec3b>(i,j)=vecf;

        }
    }
    return changeImg;
}

//均值滤波
Mat meanFilter(Mat img){
    Mat changeImg(img.rows,img.cols,CV_8UC3);

    for(int i=0;i<img.rows;i++){
        for(int j=0;j<img.cols;j++){
            int x=i;
            int y=j;
            if(x<=0){
                x=1;
            }
            if(x>=img.rows-1){
                x=img.rows-2;
            }
            if(y<=0){
                y=1;
            }
            if(y>=img.cols-1){
                y=img.cols-2;
            }

            double filter[3][3]={{1/9.0,1/9.0,1/9.0},{1/9.0,1/9.0,1/9.0},{1/9.0,1/9.0,1/9.0}};

            double b=0,g=0,r=0;
            for(int m=-1;m<2;m++){
                for(int n=-1;n<2;n++){
                    b+=img.at<Vec3b>(x+m,y+n)[0]*filter[m+1][n+1];
                    g+=img.at<Vec3b>(x+m,y+n)[1]*filter[m+1][n+1];
                    r+=img.at<Vec3b>(x+m,y+n)[2]*filter[m+1][n+1];
                }
            }

            Vec3b vecf={(uchar)b,(uchar)g,(uchar)r};
            changeImg.at<Vec3b>(i,j)=vecf;

        }
    }
    return changeImg;
}

//中值滤波
Mat medianFiltering(Mat img){
    Mat img2(img.rows,img.cols,CV_8UC3);
    int imgArray[9][3];
    for(int i=0;i<img.rows;i++){
        for(int j=0;j<img.cols;j++){

            int x=i,y=j;
            if(x<=0){
                x=1;
            }
            if(x>=img.rows-1){
                x=img.rows-2;
            }
            if(y<=0){
                y=1;
            }
            if(y>=img.cols-1){
                y=img.cols-2;
            }


            int k=0;
            for(int s=-1;s<2;s++){
                for(int z=-1;z<2;z++){
                    imgArray[k][0]=img.at<Vec3b>(x+s,y+z)[0];
                    imgArray[k][1]=img.at<Vec3b>(x+s,y+z)[1];
                    imgArray[k][2]=img.at<Vec3b>(x+s,y+z)[2];

                    k++;
                }
            }

            for(int i=0;i<3;i++){
                for(int j=0;j<8;j++){
                    int min=j;
                    for(int z=j;z<9;z++){
                        if(imgArray[z][i]<imgArray[min][i]){
                            min=z;
                        }
                    }
                    if(min!=j){
                        int t=imgArray[min][i];
                        imgArray[min][i]=imgArray[j][i];
                        imgArray[j][i]=t;
                    }
                }
            }


            Vec3b temp={imgArray[4][0],imgArray[4][1],imgArray[4][2]};
            img2.at<Vec3b>(x,y)=temp;

        }
    }


    return img2;

}

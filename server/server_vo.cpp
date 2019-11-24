
/**
 * @file server_vo.cpp
 * @brief 使用客户端
 * @details 细节
 * @mainpage 工程概览
 * @author dzqiu
 * @email dzqiu@hotmail.com
 * @version 1.0
 * @date 2019/11/20
 */

#include <iostream>
#include <stdlib.h>
#include <mutex>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <thread>
#include <fstream>
#include <boost/format.hpp>
#include <time.h>

#include <opencv2/xfeatures2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <sophus/se3.h>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;





#define RESIZE_RATION Size(640,480)

///特征点匹配对 kp_l左相机特征点，kp_r右相机特征点，上一帧kp_p左相机特征点，
vector<KeyPoint> kp_l,kp_r,kp_p;
vector<DMatch> match;
vector<unsigned char> buffer;
mutex m_buf;



///kitti squence 01 焦距与光心
const float focal = 718.856;
const Point2f pp = Point2f(607.1928, 185.2157);
///kitti squence 03 焦距与光心
//const float focal = 721.5377;
//const Point2f pp = Point2f(609.5593, 172.854);
///kitti 相机基线
const float baseline = 0.53715;
///kitti 相机内参
const Mat K = (Mat_<double>(3,3)<<focal, 0.0f, pp.x, 0.0f, focal, pp.y, 0.0f, 0.0f, 1.0f);

///双目相机内参变换，用于三角化
Mat K_l,K_r;

///Ground True旋转矩阵与平移向量
vector<Mat> R_gt,t_gt;


//PGA客户端处理图像尺寸为640*480,原始图像尺寸为1241*376
float width_ratio  = 1241.0/640;
float height_ratio = 376.0/480;
//float width_ratio  = 1;
//float height_ratio = 1;

#define BUF_SIZE 1024*8
#define START_FRAME 0

#define IMAGE_PATH "/home/dzqiu/backup/00/"
#define GROUND_TRUE "/home/dzqiu/backup/poses/00.txt"

/**
 * @brief   将旋转、平移矩阵转化为变换矩阵
 * @param   R 3x3旋转矩阵 t 1x3平移矩阵
 * @returns 变换矩阵4x4
 *
 */
Mat convertTMatrix(Mat R,Mat t)
{
    Mat T34,T44;
    hconcat(R,t,T34);
    Mat tmp = (Mat_<double>(1,4)<<0,0,0,1);
    vconcat(T34,tmp,T44);
    return T44;
}

/**
 * @brief   保存FPGA客户端匹配的结果
 */
void saveMatch()
{
    static int frame_count=0;

    char name[30];
    sprintf(name,"./points/kp%06d.txt",frame_count);
    ofstream OutFile;
    OutFile.open(name);
//    OutFile << fixed;
//    printf("size of kp %d\n",kp_l.size());
    for(int i=0;i<kp_l.size();i++)
    {
        if(kp_l[i].pt.x == 0 || kp_l[i].pt.y == 0 || kp_r[i].pt.x == 0||
           kp_r[i].pt.y == 0 || kp_p[i].pt.x == 0 || kp_p[i].pt.y == 0)
            continue;
        OutFile << kp_l[i].pt.x<<" " << kp_l[i].pt.y<<" ";
        OutFile << kp_r[i].pt.x<<" " << kp_r[i].pt.y<<" ";
        OutFile << kp_p[i].pt.x<<" " << kp_p[i].pt.y<<" ";
        OutFile << endl;
    }
    frame_count++;
    OutFile.close();

}

/**
 * @brief 读取kitti数据集Ground True并存在R_gt,t_gt;
 * @return 无
 */
void getPose()
{
    ifstream in(GROUND_TRUE);
    double tmp[12];
    string line;
    while(getline(in,line))
    {
        std::istringstream str(line);
        for(int i=0;i<12;i++)
            str>>tmp[i];
        Mat R = (Mat_<double>(3,3)<<tmp[0], tmp[1], tmp[2],
                                    tmp[4], tmp[5], tmp[6],
                                    tmp[8], tmp[9], tmp[10]);

        Mat t = (Mat_<double>(3,1)<<tmp[3],
                                    tmp[7],
                                    tmp[11]);

        R_gt.push_back(R);
        t_gt.push_back(t);
    }
    printf("get ground true:%d\n",R_gt.size());

}
/**
 * @brief   保存估计位姿以验证
 */
void savePose(Mat R,Mat t)
{
    ofstream out;
    R=R.inv();
    out.open("my00.txt",ios_base::app);
    Mat r1_vec;
    t=-t;
//    cv::Rodrigues(r1_vec,R);
    Mat T = convertTMatrix(R,t);
    for(int x=0;x<3;x++)
    {
        for(int y=0;y<4;y++)
        {
            out<<T.at<double>(x,y);
            if(!(x==2 && y==3)) out<<" ";
        }
    }
    out<<endl;
    out.close();

}

/**
 * @brief SURF特征点检测与描述、匹配，用于模拟fpga的功能
 *
 * @param img_l,img_r,img_p 对应当前时刻左右图像与上一时刻左相机图像
 * @param nkp_l,nkpr,nkp_p 对应
 * @return 返回说明
 */
void SURF_Detection(Mat img_l,Mat img_r,Mat img_p,vector<KeyPoint> &nkp_l,vector<KeyPoint> &nkp_r,vector<KeyPoint> &nkp_p)
{
    nkp_l.clear();nkp_r.clear();nkp_p.clear();
    int minHessian = 400;
    Ptr<SURF> detector = SURF::create(minHessian, 4, 3, true, true);
    vector<KeyPoint> tkp_l,tkp_r,tkp_p;
    Mat tdes_l,tdes_r,tdes_p;
    detector->detectAndCompute(img_l, Mat(), tkp_l, tdes_l);
    detector->detectAndCompute(img_r, Mat(), tkp_r, tdes_r);
    detector->detectAndCompute(img_p, Mat(), tkp_p, tdes_p);
    FlannBasedMatcher matcher;
    vector<DMatch> matches_lr,matches_lp;
    matcher.match(tdes_l, tdes_r, matches_lr);
    matcher.match(tdes_l, tdes_p, matches_lp);
    double minDist_lr = 1000000;
    double minDist_lp = 1000000;


    for(int i=0;i<matches_lr.size();i++)
        if(minDist_lr>matches_lr[i].distance)
            minDist_lr = matches_lr[i].distance;
    for(int i=0;i<matches_lp.size();i++)
        if(minDist_lp>matches_lp[i].distance)
            minDist_lp = matches_lp[i].distance;


    for(int i=0;i<matches_lr.size();i++)
    {
        if(matches_lr[i].distance>max(minDist_lr*3,0.02)) continue;
        int l = matches_lr[i].queryIdx;
        int r = matches_lr[i].trainIdx;

        for(int j=0;j<matches_lp.size();i++)
        {
            if(matches_lp[j].distance>max(minDist_lp*3,0.02)) continue;
            if(matches_lp[j].queryIdx!=i) continue;
            int p = matches_lp[j].trainIdx;
            nkp_l.push_back(tkp_l[l]);
            nkp_r.push_back(tkp_r[r]);
            nkp_p.push_back(tkp_p[p]);
            break;
        }
    }
//    printf("get good matche %d\n",nkp_l.size());
}


/**
 * @brief 根据FPGA通信协议将收到的匹配点包转化为坐标
 * @param   buffer为收到的数据包
 *          数据包大小为1024*8，最多1024对特征点，每个坐标位宽为10，共10*6=60bit,8字节中后4位保留。
 */
void buffer2points(vector<unsigned char> buffer)
{
    m_buf.lock();
    int kp_counter=0;
    kp_l.clear();kp_p.clear();kp_r.clear();match.clear();
    for(int i=0;i<BUF_SIZE/8;i++)
    {
        unsigned char bd[8];
        for(int j=0;j<8;j++)
            bd[j] = buffer[i*8+j];

#if 1
        double ly = (  bd[0]             + ((bd[1]&0x07)<<8));//11bits
        double lx = (((bd[1] & 0xf8)>>3) + ((bd[2]&0x1f)<<5));
        double py = (((bd[2] & 0xe0)>>5) + ((bd[3]&0xff)<<3));//11bits
        double px = (  bd[4]             + ((bd[5]&0x03)<<8));
        double ry = (((bd[5] & 0xfc)>>2) + ((bd[6]&0x1f)<<6)); //11bits
        double rx = (((bd[6] & 0xe0)>>5) + ((bd[7]&0x7f)<<3));
#else
        double ly = (  bd[0]             + ((bd[1]&0x03)<<8))*height_ratio;
        double lx = (((bd[1] & 0xfc)>>2) + ((bd[2]&0x0f)<<6))*width_ratio;
        double py = (((bd[2] & 0xf0)>>4) + ((bd[3]&0x3f)<<4))*height_ratio;
        double px = (((bd[3] & 0x03)>>6) + ((bd[4]&0xff)<<2))*width_ratio;
        double ry = (((bd[5] & 0xff)>>0) + ((bd[6]&0x03)<<8))*height_ratio;
        double rx = (((bd[6] & 0xfc)>>2) + ((bd[7]&0x0f)<<6))*width_ratio;
#endif
        if(ly != 0)
        {
//            printf("%d %d\n",lx,ly);
            kp_l.push_back(KeyPoint(lx,ly,2));
            kp_p.push_back(KeyPoint(px,py,2));
            kp_r.push_back(KeyPoint(rx,ry,2));
            match.push_back(DMatch(kp_counter,kp_counter,0));
            kp_counter++;
        }
    }
    saveMatch();
    m_buf.unlock();
    printf("get match points:%d\n",match.size());
}

/**
 * @brief 将特征点转化为2×N矩阵，用于是三角化
 * @param kp为为关键点数据
 * @returns 2×N矩阵
 */
Mat Vector2Mat(vector<KeyPoint> kp)
{
//    printf("start convert\n");
    Mat pm=(Mat_<float>(2,1)<<kp[0].pt.x,kp[0].pt.y);
    for(int i=1;i<kp.size();i++)
    {
         Mat p=(Mat_<float>(2,1)<<kp[i].pt.x,kp[i].pt.y);
         Mat tmp;
         hconcat(pm,p,tmp);
         pm = tmp.clone();
    }
//    printf("convert %d kp to Mat (%d %d)\n",kp.size(),pm.rows,pm.cols);
    return pm;
}
/**
 * @brief 将旋转、平移矩阵转化为李代数SE3
 * @param   R 3x3旋转矩阵 t 1x3平移矩阵
 * @returns 李代数se3
 *
 */
Sophus::SE3 Mat2SE3(Mat R,Mat t)
{

    Mat r;
    Rodrigues(R,r);

    Sophus::SE3 se3 = Sophus::SE3(
                Sophus::SO3(r.at<double>(0,0), r.at<double>(1,0), r.at<double>(2,0)),
                Eigen::Vector3d( t.at<double>(0,0), t.at<double>(1,0), t.at<double>(2,0)));
    return se3;
}


/**
 * @brief   处理线程
 * @details 主要解算匹配点对，使用三角化或视差估计特征点深度，
 *          然后使用RANSAC提出pnp的ouliers，最后使用PNP求出两帧之间的位置与姿态
 *
 */

void process()
{

    vector<KeyPoint> tkp_l,tkp_r,tkp_p;
    vector<DMatch> tmatch;



    Mat r;
    Rodrigues(R_gt[START_FRAME],r);

    //!Ground True起始点 位姿
    Sophus::SE3 T_gt= Sophus::SE3(
                Sophus::SO3(r.at<double>(0,0), r.at<double>(1,0), r.at<double>(2,0)),
                Eigen::Vector3d(t_gt[0].at<double>(0,0), t_gt[0].at<double>(1,0), t_gt[0].at<double>(2,0)));
    //!Ground Ture起始点归一化坐标（X,Y,Z,1）
    Mat gt_v4=(Mat_<double>(4,1)<<t_gt[0].at<double>(0,0), t_gt[0].at<double>(1,0), t_gt[0].at<double>(2,0),1);

    char tail_name[20];
    int numFrame=START_FRAME;
    //!估计起始位置姿态，使用Ground True指定起始点
    Sophus::SE3 T_cur=T_gt;
//    Sophus::SE3 T_cur = Sophus::SE3(Eigen::Matrix3d::Identity(),
//                                    Eigen::Vector3d::Zero());

    printf("start process thread\n");

    Mat traj(621,621,CV_8UC3);

    Sophus::SE3 Last_ref_from_cur;
    clock_t start = clock();
    while(1)
    {
         m_buf.lock();
         if(kp_l.size()>0)
         {
             tkp_l = kp_l;tkp_r=kp_r;tkp_p=kp_p;tmatch = match;
             kp_l.clear();kp_r.clear();kp_p.clear();match.clear();
             m_buf.unlock();
         }
         else
         {
             m_buf.unlock();
//             usleep(20);
             continue;
         }
         vector<Point3f> kp_3d;
         vector<Point2f> kp_2d;
#if 0
        //！双目三角化估计深度
        Mat l_mat = Vector2Mat(tkp_l);
        Mat r_mat = Vector2Mat(tkp_r);
        Mat mat4d;
        printf("start triangulate points\n");
        triangulatePoints(K_l,K_r,l_mat,r_mat,mat4d);

        for (int i=0;i<mat4d.cols;i++)
        {
            Point3f p = Point3f(mat4d.at<double>(0,i),mat4d.at<double>(1,i),mat4d.at<double>(2,i));
            p = p / mat4d.at<double>(3,i);

//            if(p.z>0)
            {
             kp_3d.push_back(p);
             kp_2d.push_back(Point2f(tkp_p[i].pt.x,tkp_p[i].pt.y));
            }
        }
        printf("kp 3d  %d\n",kp_3d.size());

#else
        printf("***************************Frame %03d***************************\n",numFrame);

        //!双目视差估计深度
        for (int i=0;i<tkp_l.size();i++)
        {
            float diff=tkp_l[i].pt.x-tkp_r[i].pt.x;
            if(diff>0.5 && diff<150)
            {
                float b_by_d = baseline / diff;
                float Z = focal * b_by_d;
                float X = (tkp_l[i].pt.x-pp.x) * b_by_d;
                float Y = (tkp_l[i].pt.y-pp.y) * b_by_d;
                kp_3d.push_back(Point3f(X,Y,Z));
                kp_2d.push_back(Point2f(tkp_p[i].pt.x,tkp_p[i].pt.y));
            }
        }
//        printf("kp 3d  %d\n",kp_3d.size());

#endif
        //!显示左右帧图像匹配结果
        sprintf(tail_name, "%06d.png", numFrame);
        Mat img_l = imread(string(IMAGE_PATH)+"image_0/"+tail_name, 0);
        Mat img_r = imread(string(IMAGE_PATH)+"image_1/"+tail_name, 0);
        Mat Match_lr,Match_lp;
        drawMatches(img_l,tkp_l,img_r,tkp_r,tmatch,Match_lr);
        resize(Match_lr,Match_lr,Size(Match_lr.cols/2,Match_lr.rows/2));
        imshow("image_left",Match_lr);
        waitKey(5);
        numFrame++;

        ///!如果三角化点不足4对，则无法使用pnp求解
        if(kp_3d.size()<=4)
        {
            usleep(50);
            continue;
        }


        //!使用PNPRANSAC找出outlier
        Mat rvec,tvec;
        vector<int> inliers;
        solvePnPRansac(kp_3d,kp_2d,K,Mat(),rvec, tvec, false, 500, 2.0f, 0.999, inliers, SOLVEPNP_ITERATIVE);
//        printf("inliers:%d\n",inliers.size());


        double inliers_ratio=1.0*inliers.size()/kp_3d.size();
        //!提出outlier并使用pnp求解位姿
        vector<Point3f> pts3d;
        vector<Point2f> pts2d;
        for(int i=0;i<inliers.size();i++)
        {
            pts3d.push_back(kp_3d[inliers[i]]);
            pts2d.push_back(kp_2d[inliers[i]]);
        }
//        printf("chosse inlier %d to slove pnp\n",pts3d.size());


         Sophus::SE3 T_ref_from_cur;
        if(pts3d.size()>=5 && inliers_ratio>0.30)
        {
            solvePnP(pts3d, pts2d, K, Mat(), rvec, tvec);
            //!将两帧相对位姿累计，计算相对于出发点的位姿
            T_ref_from_cur = Sophus::SE3(
                        Sophus::SO3(rvec.at<double>(0,0), rvec.at<double>(1,0), rvec.at<double>(2,0)),
                        Eigen::Vector3d( tvec.at<double>(0,0), tvec.at<double>(1,0), tvec.at<double>(2,0)));
            if((tvec.at<double>(0,0)*tvec.at<double>(0,0)
                + tvec.at<double>(1,0)*tvec.at<double>(1,0)
                + tvec.at<double>(2,0)*tvec.at<double>(2,0))>2.0)
            {
                 T_ref_from_cur=Last_ref_from_cur;
            }

        }
        else
        {
            T_ref_from_cur=Last_ref_from_cur;
        }

        Last_ref_from_cur = T_ref_from_cur;
//        optimizePose(pts3d, pts2d,T_ref_from_cur);

        Sophus::SE3 T_cur_from_ref = T_ref_from_cur.inverse();
        T_cur = T_cur_from_ref * T_cur;
        //!通过位姿算还原当前坐标
        Mat t = (Mat_<double>(3,1)<<T_cur.translation()[0],
                T_cur.translation()[1],
                T_cur.translation()[2]);
        Mat r =  (Mat_<double>(3,3)<<T_cur.rotation_matrix()(0,0),T_cur.rotation_matrix()(0,1),T_cur.rotation_matrix()(0,2),
                T_cur.rotation_matrix()(1,0),T_cur.rotation_matrix()(1,1),T_cur.rotation_matrix()(1,2),
                T_cur.rotation_matrix()(2,0),T_cur.rotation_matrix()(2,1),T_cur.rotation_matrix()(2,2));

        t = r.inv()*t;
        cout<<"****************"<<endl;
        cout<<r<<endl<<t<<endl;
//        cout<<"rotation angle"<<r1_vec<<endl;

        int x=-t.at<double>(0,0)+400;
        int y= t.at<double>(2,0)+500;
        circle(traj,Point(x,y),2,Scalar(0,255,0),-1);
        cout<<"estimate pose:"<<-t.at<double>(0,0)<<" " <<t.at<double>(2,0)<<endl;
        string est_text=boost::str(boost::format("estimate pose  :x=%.03f y=%.03f") % t.at<double>(0,0) % t.at<double>(2,0));
        savePose(r,t);



        //!描出Ground True进行参考
        if(numFrame!=START_FRAME)
        {
              Mat ref = convertTMatrix(R_gt[numFrame-1],t_gt[numFrame-1]);
              Mat cur = convertTMatrix(R_gt[numFrame],t_gt[numFrame]);
              cout<<"_____________"<<endl;
              cout<<cur<<endl;
              Mat r2_vec;
              cv::Rodrigues(R_gt[numFrame],r2_vec);
              cout<<"gt rotation angle"<<r2_vec<<endl;
              Mat T_true = cur*ref.inv();
              gt_v4 = T_true*gt_v4;
              x = gt_v4.at<double>(0,0)+400;
              y = -gt_v4.at<double>(2,0)+500;
              circle(traj,Point(x,y),1,Scalar(255,255,255),-1);
              cout<<"groud true pose:"<< gt_v4.at<double>(0,0)<<" "<< -gt_v4.at<double>(2,0) <<endl;
        }


        Mat res_show(850,1241,CV_8UC3);
        res_show.setTo(cv::Scalar(0, 0, 0));
        traj.copyTo(res_show(Rect(0,0,traj.cols,traj.rows)));
        resize(img_l,img_l,Size(620,188));
        cvtColor(img_l,img_l,CV_GRAY2BGR);

        img_l.copyTo(res_show(Rect(621,300,img_l.cols,img_l.rows)));
        int baseline;
        Size text_size = cv::getTextSize("left camera view", FONT_HERSHEY_COMPLEX, 1.2, 1, &baseline);
        putText(res_show,"left camera view",Point(930-text_size.width/2,250+text_size.height/2),FONT_HERSHEY_COMPLEX,1.2,Scalar(255,0,255),1);


        Match_lr.copyTo(res_show(Rect(0,650,Match_lr.cols,Match_lr.rows)));
        string gt_text=boost::str(boost::format("ground true    :x=%.03f y=%.03f")
                                  % -gt_v4.at<double>(0,0)
                                  % -gt_v4.at<double>(2,0));
        putText(res_show,gt_text,Point(700,50),FONT_HERSHEY_COMPLEX,0.8,Scalar(255,255,255),1);
        putText(res_show,est_text,Point(700,80),FONT_HERSHEY_COMPLEX,0.8,Scalar(0,255,0),1);
        string err_text=boost::str(boost::format("absolute error  :x=%.03f y=%.03f")
                                  % fabs(t.at<double>(0,0)+gt_v4.at<double>(0,0))
                                  % fabs(t.at<double>(2,0)+gt_v4.at<double>(2,0)));
        putText(res_show,err_text,Point(700,110),FONT_HERSHEY_COMPLEX,0.8,Scalar(0,0,255),1);

        clock_t end = clock();
        string cost_text=boost::str(boost::format("calculate pose time:%.02f ms")
                                  %((double)(end-start)*1000/CLOCKS_PER_SEC));

        putText(res_show,cost_text,Point(700,150),FONT_HERSHEY_COMPLEX,0.8,Scalar(255,0,255),1);

        text_size = cv::getTextSize("Match Keypoints Between Left-Right Camera", FONT_HERSHEY_COMPLEX, 1.2, 1, &baseline);
        putText(res_show,"Match Keypoints Between Left-Right Camera",Point(900-text_size.width/2,621+text_size.height/2),FONT_HERSHEY_COMPLEX,0.8,Scalar(255,0,255),1);
        start=clock();
        imshow("Result",res_show);
//        usleep(20);
    }
    printf("end of the thread\n");
}


/**
 * @brief   buffer缓冲
 * @param   buf缓冲指针，len缓冲数据长度
 *
 */

int push_buffer(unsigned char *buf,int len)
{
    for(int i=0;i<len;i++)
    {
        buffer.push_back(buf[i]);
        //!buffer足够1024*8
        if(buffer.size()==BUF_SIZE)
        {
            buffer2points(buffer);
            buffer.clear();
        }
    }

}

int main(int argc, char *argv[])
{

    printf("start system...\n \r");
    //导入ground true
    getPose();

    //双目相机内参
    Mat l_element;
    Mat r_element;
    Mat tmp =(Mat_<double>(3,1)<<0,0,0);
    hconcat(Mat::eye(3,3,CV_64F),tmp,l_element);


    tmp = (Mat_<double>(3,1)<< -baseline,0,0);

    hconcat( Mat::eye(3,3,CV_64F),tmp,r_element);
    K_l = K * l_element;
    K_r = K * r_element;
//    cout<<K_l<<endl;
//    cout<<K_r<<endl;
    //!创建TCP服务器，并监听端口
    int serv_socket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    struct sockaddr_in serv_addr;
    memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = inet_addr("192.168.1.107");
    serv_addr.sin_port = htons(7778);

    //bind  server socket and server addr
    bind(serv_socket, (struct sockaddr*)&serv_addr, sizeof(serv_addr));
    listen(serv_socket, 30);
    struct sockaddr_in clnt_addr;
    socklen_t clnt_addr_size = sizeof(clnt_addr);
    printf("wait for client login in \n");
    int clnt_socket = accept(serv_socket, (struct sockaddr*)&clnt_addr,
                        (socklen_t *)&clnt_addr_size );
    printf("client login in\n");
    unsigned char sendbuffer[BUF_SIZE] = {0};
    unsigned char recvbuffer[BUF_SIZE] = {0};

    thread multiprocess{process};
    while(1){
        int len = read(clnt_socket, recvbuffer, BUF_SIZE);
        push_buffer(recvbuffer,len);
        memset(&recvbuffer, 0, BUF_SIZE);
//        printf("rev:%d\n",len);
    }

    close(clnt_socket);
    close(serv_socket);

    return 0;
}

clear
clc
tic
%通过小样本POD来预测大样本的数据，并进行误差计算
CurveType='耗竭曲线';% 耗竭曲线 出口温度曲线
U_excel=xlsread('脱附-开式-冷凝温度0-变量hvTq版本4.xlsx',CurveType);% 训练集   耗竭曲线  变量hvTq版本5 变量hvTq版本4 变量hvTq非均匀分布625样本寝室计算
sample_excel=xlsread('样本点xn-hvTq.xlsx','625样本'); %,'增加样本量' 750新样本  625样本 625非均匀布点
%%
h_range=[0.1 0.3 0.5 0.7 0.9];%[0.1 0.2 0.3 0.7 0.9]
v_range=[0.1 0.2 0.3 0.4 0.5]; %[0.1 0.15 0.2 0.3 0.4 0.5]; %[0.1 0.2 0.3 0.4 0.5] [0.1 0.15 0.2 0.3 0.5]625样本中没有0.15这个数，750样本中有0.15这个数
T_range=80:40:240;
q_range=[1.6 1.8 2 2.2 2.35];% [1.6 1.7 1.8 2 2.35];
[x,y,z,p]=ndgrid(h_range,v_range,q_range,T_range);
% [x,y,z,p]=ndgrid(T_range,q_range,v_range ,h_range );

X=reshape(x,[],1);
Y=reshape(y,[],1);
Z=reshape(z,[],1);
P=reshape(p,[],1);
sample=[X,Y,Z,P];
RearrangeNumber=zeros(length(X),1);
U=zeros(size(U_excel));
sample = round(sample, 3);
sample_excel = round(sample_excel, 3);
for i=1:length(X)
%     RearrangeNumber(i,1) = find(ismember(sample, sample_excel(i,:), 'rows'));
RearrangeNumber(i,1) = find(ismember(sample_excel, sample(i,:), 'rows'));
U(:,i)=U_excel(:,RearrangeNumber(i,1));
i
end

num_mode=500;%设置模态数  40 500
num_sample=size(U,2);%设置样本数
% U=double(U);
U_mean=mean(U,2);%如果按照python中的写法，这行就是 U_mean=mean(U,1); mean(U,2)
U_pulsating=U-U_mean;
UTU=U_pulsating'*U_pulsating;

%求特征值并排序
[eigen_vector,eigen_value]=eig(UTU);
eigen_value=diag(eigen_value)';
% eigen_value1=eigen_value1';
eigen_vector=fliplr(eigen_vector);
eigen_value=fliplr(eigen_value);
% eigen_vector=eigen_vector(:,1:num_mode);
% eigen_value=eigen_value(1:num_mode);

%求模态
mode=zeros(size(U));
% for k=1:num_sample
%     Temporary_Matrix=zeros(size(U));
%     for i=1:num_sample
%         Temporary_Matrix(:,i)=U_pulsating(:,i)*eigen_vector(k,i);
%     end
%     Temporary_Vector=sum(Temporary_Matrix,2);
%     mode(:,k)=Temporary_Vector/norm(Temporary_Vector);
% end
for i=1:num_sample
    Temporary_Matrix=zeros(size(U));
    for n=1:num_sample
        Temporary_Matrix(:,n)=U_pulsating(:,n)*eigen_vector(n,i);
    end
    Temporary_Vector=sum(Temporary_Matrix,2);
    mode(:,i)=Temporary_Vector/norm(Temporary_Vector);
end

mode=mode(:,1:num_mode);

%求投影系数
projection_coefficient=zeros(num_mode,num_sample);
for i=1:num_mode
    for j=1:num_sample
        projection_coefficient(i,j)=U_pulsating(:,j)'*mode(:,i)/(norm(mode(:,i)))^2;
    end
end

%%
% POD重构

%%%%%%%%%%%%%%%%%%%%%%%%%%
%随机测试集样本
U_test=xlsread('脱附-开式-冷凝温度0-变量hvTq-随机样本.xlsx',CurveType);%测试集曲线 
sample_test=xlsread('样本点xn-随机样本.xlsx','hvTq');%测试集样本

%%%%%%%%%%%%%%%%%%%%%%
%把多个表格中的数值整合到一起,搭建测试集的样本和曲线
q=1.6:0.1:2.3;
UU=zeros(2881,1);
sample=zeros(1,4);
for i=1:size(q,2)

filename = sprintf('脱附-开式-冷凝温度0-变量hvT-q%.1f.xlsx', q(i));
depletion_curve=xlsread(filename,CurveType);
UU=[UU depletion_curve];
% temp_outlet_curve=xlsread(filename,'出口温度曲线');

yangben=xlsread('样本点xn-hvT.xlsx','729样本q2.2'); %,'增加样本量'
if q(i)==2.3
    q(i)=2.35;
end
q_i=q(i)*ones(size(yangben,1),1);
yangben=[yangben(:,1) yangben(:,2) q_i yangben(:,3)];
sample=[sample;yangben];
    i
end
U_test=UU(:,2:end);%测试集曲线 
sample_test=sample(2:end,:);%测试集样本
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
condition=sample_test';

% [xi,yi,zi] = meshgrid(0:10:350, 5:5:60, 0.5:0.5:1.5);
xishu=projection_coefficient;
% sample=xlsread('样本点xn-hvTq.xlsx','625样本'); %,'增加样本量'
sim_data=U;% U
mean_C_distribution=mean(sim_data,2);

num_motai=size(xishu,1);
% xishu_new=zeros(num_motai,size(xi,2)*size(yi,2)*size(zi,2));
for i=1:num_motai
    v=xishu(i,:);



%griddedInterpolant,基于网格数据的插值方法
V=reshape(v,[length(h_range) length(v_range) length(q_range) length(T_range)]);
F = griddedInterpolant(x,y,z,p,V,'makima');% 指定插值方法：'linear'、'nearest'、'next'、'previous'、'pchip'、'cubic'、'makima' 或 'spline'
xinxishu=F(sample_test(:,1),sample_test(:,2),sample_test(:,3),sample_test(:,4));
xinxishu = reshape(xinxishu,1,[]);
xishu_new(i,:)=xinxishu;



i
end
%重构浓度场
motai=mode;
C_distribution_new=motai*xishu_new+mean_C_distribution;
C_sample_yanzheng=motai*projection_coefficient+mean_C_distribution;
chazhidian=condition';

% U_test;C_distribution_new;峰值高度，峰值位置，累计脱附量（500和1000分钟） ，rmse（500和1000分钟）
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
results=zeros(size(sample_test,1),18);
results=zeros(size(sample_test,1),20);   %%%%%%%%%%计算温度误差的时候使用到%%%%%%%%%%%%
for i=1:size(C_distribution_new,2)
curve_sim=U_test(:,i);
curve_POD=C_distribution_new(:,i);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[C1,T1]=max(curve_sim(:,1));%找到峰值时间
[C2,T2]=max(curve_POD(:,1));%找到峰值时间
ERROR_C=(C1-C2)/C1*100;%浓度相对误差
ERROR_T=(T1-T2)/T1*100;%时间相对误差，不过这个分辨率不够其实
integration1_500=trapz(1:500,curve_sim(1:500,1));%脱附量面积积分
integration1_1000=trapz(1:1000,curve_sim(1:1000,1));%脱附量面积积分
integration2_500=trapz(1:500,curve_POD(1:500,1));%脱附量面积积分
integration2_1000=trapz(1:1000,curve_POD(1:1000,1));%脱附量面积积分
ERROR_q_500=(integration1_500-integration2_500)/integration1_500*100;
ERROR_q_1000=(integration1_1000-integration2_1000)/integration1_1000*100;
RMSE_500=sqrt(mean((curve_sim(1:500,1)-curve_POD(1:500,1)).^2));
RMSE_1000=sqrt(mean((curve_sim(1:1000,1)-curve_POD(1:1000,1)).^2));
R2_500=1-sum((curve_sim(1:500,1)-curve_POD(1:500,1)).^2)/sum((curve_sim(1:500,1)-mean(curve_sim(1:500,1))).^2);
R2_1000=1-sum((curve_sim(1:1000,1)-curve_POD(1:1000,1)).^2)/sum((curve_sim(1:1000,1)-mean(curve_sim(1:1000,1))).^2);
pearson_r_500=(500*sum(curve_sim(1:500,1).*curve_POD(1:500,1))-sum(curve_sim(1:500,1))*sum(curve_POD(1:500,1)))...
    /((500*sum(curve_sim(1:500,1).^2)-sum(curve_sim(1:500,1))^2)^0.5*(500*sum(curve_POD(1:500,1).^2)-sum(curve_POD(1:500,1))^2)^0.5);
pearson_r_1000=(1000*sum(curve_sim(1:1000,1).*curve_POD(1:1000,1))-sum(curve_sim(1:1000,1))*sum(curve_POD(1:1000,1)))...
    /((1000*sum(curve_sim(1:1000,1).^2)-sum(curve_sim(1:1000,1))^2)^0.5*(1000*sum(curve_POD(1:1000,1).^2)-sum(curve_POD(1:1000,1))^2)^0.5);
MAE500=mean((curve_sim(1:500,1)-curve_POD(1:500,1)));% mean absolute error %%%%%%%%%%%%%%计算温度误差的时候使用到%%%%%%%%%%%
MAE1000=mean((curve_sim(1:1000,1)-curve_POD(1:1000,1)));% mean absolute error %%%%%%%%%%%%%%计算温度误差的时候使用到%%%%%%%%%%%


results(i,1:4)=sample_test(i,1:4);%工况 h v T q
results(i,5)=C1;
results(i,6)=C2;
results(i,7)=ERROR_C;
results(i,8)=T1;
results(i,9)=T2;
results(i,10)=ERROR_T;
results(i,11)=ERROR_q_500;
results(i,12)=ERROR_q_1000;
results(i,13)=RMSE_500;
results(i,14)=RMSE_1000;
results(i,15)=R2_500;
results(i,16)=R2_1000;
results(i,17)=pearson_r_500;
results(i,18)=pearson_r_1000;
results(i,19)=MAE500;   %%%%%%%%%%%%%%计算温度误差的时候使用到%%%%%%%%%%%
results(i,20)=MAE1000; %%%%%%%%%%%%计算温度误差的时候使用到%%%%%%%%%%%%
i
end
toc
% [x,y]=find(C_distribution_new==0.00358603375553747)
% condition_POD=condition(:,1081423) 
% y_POD=C_distribution_new(:,[315981 672342 1081423]);672342 315981
% xlswrite('POD误差计算.xlsx',results,'40模态900样本griddatan函数')


% point=1636;
% condition_POD=condition(:,point) 
% y_POD=C_distribution_new(:,point);
% y_sim=U_test(:,point);
% C_compare=[y_sim y_POD];
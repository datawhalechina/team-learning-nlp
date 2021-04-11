#本文件功能：将通过python脚本进行音频增强后的数据进行处理，用sox进行规整成16bit，16000的采样率的wav文件。以下是批量操作。   

#! /bin/bash
function read_dir(){
for file in `ls $1` #注意此处这是两个反引号，表示运行系统命令
do
   if [ -d $1"/"$file ] #注意此处之间一定要加上空格，否则会报错
   then
       read_dir $1"/"$file
   else
   #echo $1"/"$file #在此处处理文件即可
       new_dir="a/"$1
       if [ ! -d $new_dir  ];then
           mkdir -p $new_dir    #没有文件夹则创建
       fi
   sox $1"/"$file -b 16 -e signed-integer $new_dir"/"$file
   fi
done
} 
#读取第一个参数 
read_dir clips_rd

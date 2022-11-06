#!/usr/bin/bash
#FileList=($(ls  /data/mqzhang/aux_data/OCO2/2019/oco2*.nc4)) 
#FileList=($(ls  /data/mqzhang/aux_data/OCO2/2019/oco2_LtCO2_190[4-9]*.nc4)) 
FileList=($(ls  /data/mqzhang/aux_data/OCO2/201[4-8]/oco2_LtCO2_*.nc4)) 
#FileList=($(ls  /home/mqzhang/DATA_STORE/CO2/GOSAT/ACOS_L2_Lite_FP.9r/2019/acos*.nc4)) 
Njob=${#FileList[*]}    # 作业数目
Nproc=30    # 可同时运行的最大作业数

function CMD {        # 测试命令, 随机等待几秒钟
    #python3 unify_ak_GOSAT.py $1
    python3 unify_ak_OCO2_New.py $1
}

PID=() # 记录PID到数组, 检查PID是否存在以确定是否运行完毕
for((i=1; i<=Njob; )); do
    for((Ijob=0; Ijob<Nproc; Ijob++)); do
        if [[ $i -gt $Njob ]]; then
            break;
        fi
        if [[ ! "${PID[Ijob]}" ]] || ! kill -0 ${PID[Ijob]} 2> /dev/null; then
            CMD ${FileList[$i]} &
            PID[Ijob]=$!
            i=$((i+1))
        fi
    done
    sleep 1
done
wait

# wxy_chiplet_code

## Structure  
- DSE : 设计空间探索的主题代码，主要涉及单网络映射与多网络映射  
  - SE_DSE : 单引擎设计空间探索，为单网络映射算法（单引擎）  
  - MN_DSE : 多网络设计空间探索  
- interconnect : 包含与互连拓扑相关的文件  
  - dynamic_topology : 基于动态可重构线模型，随机开关状态，生成对应的线  
  - gem5_topology : 生成gem5 garnet所需要的拓扑文件，便于自己去创造拓扑  
- trace_generator : 生成trace文件，包含了nn数据流trace和随机数据流trace（简单版本）  

## Command  

### DSE  
查看run.sh文件，通过修改配置信息，来运行想要的程序，注意关注路径    
```shell
sh run.sh
```


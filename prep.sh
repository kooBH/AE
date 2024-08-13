

DIR_IN="/home/data/kbh/LRS3/trainval"
DIR_OUT="/home/data/kbh/lip/LRS3/trainval"
#python src/Prep.py -i ${DIR_IN} -o ${DIR_OUT} 

DIR_IN="/home/data/kbh/LRS3/test"
DIR_OUT="/home/data/kbh/lip/LRS3/test"
python src/Prep.py -i ${DIR_IN} -o ${DIR_OUT} 


DIR_IN="/home/data/kbh/LRS2/main"
DIR_OUT="/home/data/kbh/lip/LRS2/main"
python src/Prep.py -i ${DIR_IN} -o ${DIR_OUT} 

DIR_IN="/home/data/kbh/LRS2/pretrain"
DIR_OUT="/home/data/kbh/lip/LRS2/pretrain"
python src/Prep.py -i ${DIR_IN} -o ${DIR_OUT} 

currpath=`pwd`
# train the model
python main.py --phase train --model_file ${currpath}/config/drmm_tks.mq2007.config
# predict with the model
python main.py --phase predict --model_file ${currpath}/config/drmm_tks.mq2007.config
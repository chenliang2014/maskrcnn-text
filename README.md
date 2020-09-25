# maskrcnn-text
用mask-rcnn检测倾斜文本

用tensorflow的预训练模型，在文本标注数据上重新训练一个检测文本的模型，并利用mask的最小包围框来获得文字方向。


依赖项：tensorflow及其Objectdetect项目。 https://github.com/tensorflow/models/tree/r1.13.0/research/object_detection

训练步骤如下：

1. 先安装tensorflow以及 object detection 项目，安装方法见 https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/tensorflow-1.14/install.html。



2. 安装Labelme 标注工具，见https://github.com/wkentaro/labelme

	
    
3.  从http://www.iapr-tc11.org/mediawiki/index.php/MSRA_Text_Detection_500_Database_%28MSRA-TD500%29  下载已标注数据集。
	将图片文件放入images\train,images\test目录下，运行images/gen_labelme_json.py 将gt格式转为labelme格式。
	运行labelme检查转换是否正确。 
	注意： 原始数据集中有几张图片没有标注，需要删除掉，否则后续程序会报错。
	

4. 进入mask-text目录，执行一下命令生成coco文件到annotations目录下：  
	python labelme2coco.py images\train --output annotations\train.json  
	python labelme2coco.py images\test --output annotations\test.json  
	
5. 生成tfrecord文件：  
	python create_coco_tf_record.py --logtostderr --train_image_dir=images\train --train_annotations_file=annotations\train.json --test_image_dir=images\test --test_annotations_file=annotations\test.json --output_dir=annotations  --include_masks=True  
	
6. 在annotations目录下创建一个文本文件label_map.pbtxt,内容为：  
	item {  
		id: 1  
		name: 'txt'  
	}  



7. 下载 mask_rcnn_inception_v2(http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz) 压缩包并解压到pre_trained_model。 

8. 将https://github.com/tensorflow/models/blob/master/research/object_detection/samples/configs/mask_rcnn_inception_v2_coco.config 下载到pre_trained_model目录下。

9. 修改mask_rcnn_inception_v2_coco.config 内容：  
	第10行： num_classes: 1  
	第127行: fine_tune_checkpoint: "pre-trained-model/mask_rcnn_inception_v2_coco_2018_01_28/model.ckpt"  
	第142行：input_path: "./annotations/train.record"  
	第144行：label_map_path: "annotations/label_map.pbtxt"  
	第158行：input_path: "./annotations/test.record"  
	第160行：label_map_path: "annotations/label_map.pbtxt"  
	

10. 训练模型：
	python train.py --logtostderr --train_dir=training/ --pipeline_config_path=pre-trained-model/mask_rcnn_inception_v2_coco.config
	
11. 查看训练情况：
	tensorboard --logdir=training
	
12. 输出训练模型：  
	python export_inference_graph.py --input_type image_tensor --pipeline_config_path=pre-trained-model/mask_rcnn_inception_v2_coco.config --trained_checkpoint_prefix training/model.ckpt-xxxxx --output_directory  maskrcnn-text-detect
	
13. 测试模型：
	python object-detection-test.py images/test/IMG_0667.JPG
	
	测试结果会保存到当前目录下IMG_0667_box.jpg， 深绿色为包围矩形框，红色为最小包围框，利用红色框可以得到文字方向。
	
	
14. 如果要在opencv的dnn中运行模型，需要先执行：
	python tf_text_graph_mask_rcnn.py --input maskrcnn-text-detect/frozen_inference_graph.pb  --config pre-trained-model/mask_rcnn_inception_v2_coco.config --output maskrcnn-text.phtxt
	
	将maskrcnn-text.phtxt作为配置文件传给python mask_rcnn.py 的--config 参数。
	如果直接使用mask_rcnn_inception_v2_coco.config 会报错。
	

预训练模型下载地址：https://cloud.189.cn/t/fqemu2FrAjIr


注意，该模型并没有充分训练，仅供测试。
	

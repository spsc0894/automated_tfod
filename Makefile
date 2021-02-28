setup:
	mkdir Tensorflow
	cd Tensorflow && git clone https://github.com/tensorflow/models.git
	git clone https://github.com/cocodataset/cocoapi.git

run_tfod:
	xhost +
	docker-compose up -d
stop_tfod:
	docker-compose down

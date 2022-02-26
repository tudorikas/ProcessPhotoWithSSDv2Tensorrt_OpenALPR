import jetson.inference
import jetson.utils
import traceback
import pika
import time
import json
import os
from io import StringIO, BytesIO
import numpy as np
import base64

from datetime import datetime
import argparse
import sys

"""
    File detectnetWork.py have the role to detect the car/truck in the given images
"""
class detectnetWork:
    def __init__(self):
        """Initialize and read from json Config file
                RabbitmqServer- Ip of the RabbitMqserver
	            RabbitmqQueueGet- Queue from where to get json
	            RabbitmqQueuePut- Queue for where to put json
	           	thresh- threshold of the network ssdv2 """
        with open('yolodetect.json') as json_file:
            self.JsonConfig = json.load(json_file)
        self.RabbitmqServer = self.JsonConfig['RabbitmqServer']
        self.RabbitmqQueuePut = self.JsonConfig['RabbitmqQueuePut']
        self.RabbitmqQueueGet = self.JsonConfig['RabbitmqQueueGet']
        self.thresh = self.JsonConfig['thresh']

        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.RabbitmqServer))
        self.channel = self.connection.channel()
        args = {"x-max-length": 200}
        self.channel.queue_declare(queue=self.RabbitmqQueueGet, durable=True,arguments=args)

        print("Load Weight")
        self.LoadWeight()
        self.LoadListDetection()
        print("Weight Loaded")

        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(queue=self.RabbitmqQueueGet, on_message_callback=self.callback)
        self.channel.start_consuming()

    def LoadWeight(self):
        """Load the weight for network."""
        # load the object detection network
        self.net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=self.thresh)

    def LoadListDetection(self):
        """Load List of the possible detection that the network can do."""
        with open('ssdv2classes.txt') as f:
            self.listDetections=f.read().splitlines()
            #self.listDetections = f.readlines()

    def send_to_rabbit(self, load_json):
        """Send the detections with the old JSON to RabbitMQ queue."""
        #print("send rabbit")
        self.connectionSend = pika.BlockingConnection(pika.ConnectionParameters(host=self.RabbitmqServer))
        self.channelSend = self.connectionSend.channel()
        args = {"x-max-length": 200}
        self.channelSend.queue_declare(queue=self.RabbitmqQueuePut, durable=True,arguments=args)
        self.channelSend.basic_publish(exchange='', routing_key=self.RabbitmqQueuePut, body=load_json)
        self.connectionSend.close()

    def callback(self, ch, method, properties, body):
        """Wait for a json from getFromQueue.py to start processing """
        try:
            jsonload = json.loads(body)
            img, width, height = jetson.utils.loadImageRGBA(jsonload["img"])
            detections = self.net.Detect(img, width, height)

            jsonObject={}
            detectedforJson = list()

            """Create json for future training"""
            for det in detections:
                detectionjson = {}
                detectionjson['type'] = self.listDetections[det.ClassID]
                detectionjson['confidence'] = det.Confidence
                detectionjson['Bottom'] = det.Bottom
                detectionjson['Top'] = det.Top
                detectionjson['Right'] = det.Right
                detectionjson['Left'] = det.Left
                detectionjson['Height'] = det.Height
                detectionjson['Width'] = det.Width
                detectedforJson.append(detectionjson)

            jsonObject['detections']=detectedforJson
            #jsonObject['rest'] = jsonload['rest']
            fileNameJson=jsonload['img'][:-3]
            fileNameJson+="json"

            with open(fileNameJson, 'w', encoding='utf-8') as f:
                json.dump(jsonObject, f, ensure_ascii=False, indent=4)
                f.close()

            detected=list()
            #detectionjson = {}
            #detectionjson['type'] = "truck"
            #detectionjson['confidence'] = 0.75
            #detected.append(detectionjson)
            for det in detections:
                """Append new detections to old json"""
                if self.listDetections[det.ClassID]=="car" or self.listDetections[det.ClassID]=="truck" \
                    or self.listDetections[det.ClassID]=="train" or self.listDetections[det.ClassID]=="bus":
                    detectionjson={}
                    detectionjson['type']=self.listDetections[det.ClassID]
                    detectionjson['confidence']=det.Confidence
                    detected.append(detectionjson)

            jsonsend = {}
            jsonsend["rest"] = jsonload
            jsonsend["detections"] = detected
            self.send_to_rabbit(json.dumps(jsonsend))
            ch.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as e:
            print(str(e))
            with open('Log_DetectnetWork.txt', 'a') as the_file:
                now = datetime.now()  # current date and time
                the_file.write(
                    now.strftime("%m/%d/%Y, %H:%M:%S") + " " + repr(e) + " " + traceback.format_exc() + "\n")
                the_file.close()
                ch.basic_ack(delivery_tag=method.delivery_tag)
aa=detectnetWork()
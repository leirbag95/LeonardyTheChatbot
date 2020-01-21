import sys

from flask import Flask, jsonify, request
import chatbot
from chatbot import Leonardy, Supervised

leoBot = chatbot.Leonardy()




app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def Hello():
	if request.method == 'POST':
		json_request = request.get_json()
		message = json_request["message"]
		try:
			leo_score = leoBot.get_response(message)
		except OSError as err:
			print("OS error: {0}".format(err))
		except ValueError:
			print("Could not convert data to an integer.")
		print(leo_score)
		return jsonify({'user_message':json_request["message"],'score':0.95, 'responses':["Hello my friend", "Hi there!", "Yo! How can I help you ?"], 'tag':"greeting"}), 201
	else:
		return jsonify({'about':'Hello World'})


@app.route('/multi/<int:num>', methods=['GET'])
def get_multi(num):
	return jsonify({'result': num*10})

if __name__ == '__main__':
	app.run(debug=False, threaded=False)

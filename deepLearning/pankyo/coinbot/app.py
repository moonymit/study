from flask import Flask
from flask import request
from flask import jsonify

import requests

app = Flask(__name__)

coin_map = {
  '비트코인': 'btc_krw',
  '이더리움': 'eth_krw',
  '이클': 'etc_krw',
  '리플': 'xrp_krw',
}

@app.route('/')
def hello_world():
	return 'Hello World!'

@app.route('/webhook', methods=['POST'])
def webhook():
	dataDict = request.get_json()
	coinType = dataDict['result']['parameters']['coin-type'];
	coinTypeParam = coin_map[coinType];

	requestUrl = 'https://api.korbit.co.kr/v1/ticker/detailed';
	params = {'currency_pair': coinTypeParam}
	res = requests.get(requestUrl, params=params)

	if res.status_code == 404:
		return abort(404);
	
	data = res.json()
	print(data)

	speech = ""
	speech += "최종 체결 시각: {} \n".format(str(data["timestamp"]))
	speech += "최종 체결 가격:{} \n".format(str(data["last"]))
	speech += "최우선 매수호가:{} \n".format(str(data["bid"]))
	speech += "최우선 매도호가:{} \n".format(str(data["ask"]))
	speech += "최근 24시간 저가:{} \n".format(str(data["low"]))
	speech += "최근 24시간 고가:{} \n".format(str(data["high"]))
	speech += "거래량:{:.3} \n".format(data["volume"])

	return jsonify({
		'speech': speech,
		'displayText': '시세 정보가 도착했습니다.',
	}), 200

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=80)
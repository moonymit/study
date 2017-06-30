var http=require('http'),  
    https = require('https'),
    express = require('express'),
   	fs = require('fs'),
   	bodyParser = require('body-parser'),
    request = require('request');

var app = express();  
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({
    extended: true
}));

var http_port = 80;  
var coin_map = {
  '비트코인': 'btc_krw',
  '이더리움': 'eth_krw',
  '이클': 'etc_krw',
  '리플': 'xrp_krw',
}

app.get('/', function (req, res) {
  console.log('Hello world')
  res.send('Hello World!');
});

app.post('/webhook', function (req, res) {

  var coinType = req.body.result.parameters['coin-type'];
  var coinTypeParam = coin_map[coinType];

  var requestUrl = 'https://api.korbit.co.kr/v1/ticker/detailed?currency_pair=' + coinTypeParam;
  
  request(requestUrl, function (error, response, body) {
    console.log('statusCode:', response && response.statusCode);
    console.log('data:', body);

    var data = JSON.parse(body);
    console.log(data.timestamp)

    var speech = ""
      + "최종 체결 시각:" + data["timestamp"] + "\n"
      + "최종 체결 가격:" + data["last"] + "\n"
      + "최우선 매수호가:" + data["bid"] + "\n"
      + "최우선 매도호가:" + data["ask"] + "\n"
      + "최근 24시간 저가:" + data["low"] + "\n"
      + "최근 24시간 고가:" + data["high"] + "\n"
      + "거래량:" + parseInt(data["volume"]).toFixed(3)  + "\n"

    if (error) res.status(500).send(error);
    else {
      res.status(200).json({
        'speech': speech,
        'displayText': '시세 정보가 도착했습니다.',
      })
    }
  });

});


http.createServer(app).listen(http_port, function(){  
  console.log("Http server listening on port " + http_port);
});

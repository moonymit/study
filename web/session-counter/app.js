var express = require('express')
var session = require('express-session');
var app = express()

app.set('trust proxy', 1)
app.use(session({
  secret: '2314ASF3f*ASD*F&bBQ',
  resave: false,
  saveUninitialized: true,
  // cookie: { secure: true } for https
}))


// 실제 서비스에서는 DB에다가 세션 정보를 저장해야 한다.
app.get('/count', function(req, res) {
	if (req.session.count) {
		req.session.count++;
	} else {
		req.session.count = 1;
	}
	res.send('count : ' + req.session.count);
})

app.listen(3000, function () {
	console.log('session-counter app listening on port 3000!')
})
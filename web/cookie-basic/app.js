var express = require('express')
var cookieParser = require('cookie-parser')
var app = express()

app.use(cookieParser('24ASDF34ASFGADSF#*A33'))
app.get('/count', function(req, res){
	if(req.signedCookies.count){
		var count = parseInt(req.signedCookies.count);
	} else {
		var count = 0;
	}
	count = count+1;
	res.cookie('count', count, {signed: true});
	res.send('count : ' + count);
});

app.listen(3000, function () {
	console.log('cookie-basic app listening on port 3000!')
})
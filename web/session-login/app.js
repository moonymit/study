var express = require('express')
var session = require('express-session');
var bodyParser = require('body-parser');
var app = express();

app.set('trust proxy', 1);
app.use(bodyParser.urlencoded({extended:false}));
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
});

app.get('/auth/logout', function(req, res) {
	delete req.session.displayName;
	res.redirect('/welcome');
})

app.get('/welcome', function(req, res) {
	if(req.session.displayName) {
		res.send(`
			<h1>Hello, ${req.session.displayName}</h1>
			<a href="/auth/logout">Logout</a>
		`);
	} else {
		res.send(`
			<h1>Welcome</h1>
			<a href="/auth/login">Login</a>
		`);
	}
})

app.post('/auth/login', function(req, res) {
	var user = {
		username: 'hijigoo',
		password: '1111',
		displayName: 'Hi.JiGOO'
	};
	var uname = req.body.username;
	var pwd = req.body.password;
	if(uname === user.username && pwd === user.password) {
		req.session.displayName = user.displayName;
		res.redirect('/welcome');
	} else {
		res.send('Who are you? <a href="/auth/login">Login</a>');
	}
});

app.get('/auth/login', function(req, res) {
	var output = `
		<h1>Login</h1>
		<form action="/auth/login" method="post">
			<p>
				<input type="text" name="username" placeholder="username">
			</p>
			<p>
				<input type="password" name="password" placeholder="password">
			</p>
			<p>
				<input type="submit">
			</p>
		</form>
	`;
	res.send(output);
})

app.listen(3000, function () {
	console.log('session-counter app listening on port 3000!')
})
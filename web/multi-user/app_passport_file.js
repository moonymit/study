var express = require('express')
var session = require('express-session');
var bodyParser = require('body-parser');
var FileStore = require('session-file-store')(session);
var bkfd2Password = require("pbkdf2-password");
var passport = require('passport');
var LocalStrategy = require('passport-local').Strategy;
var hasher = bkfd2Password();

var app = express();

app.set('trust proxy', 1);
app.use(bodyParser.urlencoded({extended:false}));
app.use(session({
  secret: '2314ASF3f*ASD*F&bBQ',
  resave: false,
  saveUninitialized: true,
  store: new FileStore(),
}))
app.use(passport.initialize());
app.use(passport.session());

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
	// passport가 만듬
	req.logout();
	req.session.save(function() {
		res.redirect('/welcome');
	});
})

app.get('/welcome', function(req, res) {
	// passport는 user라는 객체를 req의 멤버로 만들어 준다.
	// user객체는 deserializeUser의 done 객체의 두번재 객체가 된다.
	if(req.user && req.user.displayName) {
		res.send(`
			<h1>Hello, ${req.user.displayName}</h1>
			<a href="/auth/logout">Logout</a>
		`);
	} else {
		res.send(`
			<h1>Welcome</h1>
			<ul>
				<li><a href="/auth/login">Login</a></li>
				<li><a href="/auth/register">Register</a></li>
			</ul>
		`);
	}
});

passport.serializeUser(function(user, done) {
	// session에 use.usename(식별자)이 저장된다 
	console.log('serializeUser', user);
	done(null, user.username);
});

// 다음 접속시 user.username값이 id파라메터로 들어옴
// 파라미터 done은 serializeUser의 done과는 다름
passport.deserializeUser(function(id, done) {
	console.log('deserializeUser', id)
	for(var i = 0; i < users.length; i++) {
		var user = users[i];
		if(user.username == id) {
			// req.user 를 통해 정보를 가져올 수 있게 된다.
			return done(null, user);
		}
	}
	done('There is no user.');;
});

passport.use(new LocalStrategy(
	function(username, password, done) {
		var uname = username;
		var pwd = password;

		for(var i = 0; i < users.length; i++) {
			var user = users[i];
			if(uname === user.username) {
				return hasher({password:pwd, salt:user.salt}, function(err, pass, salt, hash) {
					if((hash === user.password)) {
						console.log('LocalStrategy', user);
						done(null, user); //sserializeUser 호출
						// req.session.displayName = user.displayName;
						// req.session.save(function() {
						// 	res.redirect('/welcome')
						// })
					} else {
						done(null, false);
						// res.send('Who are you? <a href="/auth/login">Login</a>');
					}
				})
			}
		}
		done(null, false);
		// res.send('Who are you? <a href="/auth/login">Login</a>');	
	}
));

app.post('/auth/login', 
	passport.authenticate(
		'local', 
		{ 
			successRedirect: '/welcome',
			failureRedirect: '/auth/login',
			failureFlash: false 
		}
	)
);

// app.post('/auth/login', function(req, res) {

// 	var uname = req.body.username;
// 	var pwd = req.body.password;

// 	for(var i = 0; i < users.length; i++) {
// 		var user = users[i];

// 		if(uname === user.username) {
// 			return hasher({password:pwd, salt:user.salt}, function(err, pass, salt, hash) {
// 				if((hash === user.password)) {
// 					req.session.displayName = user.displayName;
// 					req.session.save(function() {
// 						res.redirect('/welcome')
// 					})
// 				} else {
// 					res.send('Who are you? <a href="/auth/login">Login</a>');
// 				}
// 			})
// 		}
// 		// if(uname === user.username && sha256(pwd + user.salt) === user.password) {
// 		// 	req.session.displayName = user.displayName;
// 		// 	return req.session.save(function() {
// 		// 		res.redirect('/welcome');
// 		// 	})
// 		// } 
// 		res.send('Who are you? <a href="/auth/login">Login</a>');
// 	}
// });

var salt = "@#$#@dag#W#SDFAFf@";
var users = [
	{
		username: 'hijigoo',
		password: 'BNvLiJfqyhLPwu7Hb8+Rj373Ifn4p0pPdLj9Ai+X6STExzGrVGf33n1t9wpJl2JYHAPvXNCetPeLK0X39g2ZJSSFiUGnW6EAK7oNG459LbD3IuXkyPiDQYZfQ2vhAXP22ULaBE1y/xljwiVCw2WUBhT/IH+57W2rhuXsQUoIvFs=', //md5 11111
		salt: 'DUS5LzN7BGCOGPsheALQXJ0/TwqQ54Srown1Xig93ssbmtknp4w7FFcHZYzelnBRJG3Yu6b2lGTFnGaCXMN5MA==',
		displayName: 'Hi.JiGOO'
	}]

app.post('/auth/register', function(req, res) {
	hasher({password: req.body.password}, function(err, pass, salt, hash) {
		var user = {
			username: req.body.username,
			password: hash,
			salt: salt,
			displayName: req.body.displayName
		};
		users.push(user);
		req.login(user, function(err) {
			req.session.save(function() {
				res.redirect('/welcome');
			})
		});
	});
})

app.get('/auth/register', function(req, res) {
	var output = `
		<h1>Register</h1>
		<form action="/auth/register" method="post">
			<p>
				<input type="text" name="username" placeholder="username">
			</p>
			<p>
				<input type="password" name="password" placeholder="password">
			</p>
			<p>
				<input type="text" name="displayName" placeholder="displayName">
			</p>
			<p>
				<input type="submit">
			</p>
		</form>
	`;
	res.send(output);
})

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
	console.log('multi-user app listening on port 3000!')
})
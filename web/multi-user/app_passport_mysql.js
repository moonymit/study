var express = require('express')
var session = require('express-session');
var bodyParser = require('body-parser');
var MySQLStore = require('express-mysql-session')(session);
var bkfd2Password = require("pbkdf2-password");
var passport = require('passport');
var LocalStrategy = require('passport-local').Strategy;
var FacebookStrategy = require('passport-facebook').Strategy;
var hasher = bkfd2Password();
var mysql = require('mysql');
var conn = mysql.createConnection({
	host	:'localhost',
	user 	:'root',
	database:'o2'
});
conn.connect();

var app = express();

app.set('trust proxy', 1);
app.use(bodyParser.urlencoded({extended:false}));
app.use(session({
  secret: '2314ASF3f*ASD*F&bBQ',
  resave: false,
  saveUninitialized: true,
  store: new MySQLStore({
	    host: 'localhost',
	    port: 3306,
	    user: 'root',
	    database: 'o2'
	})
}));

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
	done(null, user.authId);
});

// 다음 접속시 user.authId값이 id파라메터로 들어옴
// 파라미터 done은 serializeUser의 done과는 다름
passport.deserializeUser(function(id, done) {
	console.log('deserializeUser', id)
	var sql = "SELECT * FROM users where authId=?";
	conn.query(sql, [id], function(err, results) {
		if(err) {
			console.log(err);
			done('There is no user.');
		} else {
			done(null, results[0]);
		}
	});
	// for(var i = 0; i < users.length; i++) {
	// 	var user = users[i];
	// 	if(user.authId == id) {
	// 		// req.user 를 통해 정보를 가져올 수 있게 된다.
	// 		return done(null, user);
	// 	}
	// }
	// done('There is no user.');;
});

passport.use(new LocalStrategy(
	function(username, password, done) {
		var uname = username;
		var pwd = password;
		var sql = "SELECT * FROM users WHERE authId=?";
		conn.query(sql, ['local:' + uname], function(err, results) {
			console.log(results);
			if (err) {
				return done('There is no user.');
			}

			var user = results[0];
			return hasher({password:pwd, salt:user.salt}, function(err, pass, salt, hash) {
				if((hash === user.password)) {
					console.log('LocalStrategy', user);
					done(null, user);
				} else {
					done(null, false);
				}
			})
		});
	}
));

passport.use(new FacebookStrategy({
		clientID: '217231225370668',
		clientSecret: 'ce2a0fc172636a49ebe12628d86bf15e',
		callbackURL: "/auth/facebook/callback",
		profileFields: ['id', 'email', 'displayName']
	},
	function(accessToken, refreshToken, profile, done) {
		console.log(profile);
		var authId = 'facebook:' + profile.id;
		var sql = "SELECT * FROM users WHERE authId=?";
		conn.query(sql, [authId], function(err, results) {
			if(results.length > 0 ) {
				done(null, results[0]);
			} else {
				var newuser = {
					'authId': authId,
					'displayName': profile.displayName,
					'email': profile.emails[0].value
				};
				var sql = "INSERT INTO users SET ?";
				conn.query(sql, newuser, function(err, results) {
					if(err) {
						console.log(err);
						done('Error');
					} else {
						done(null, newuser);
					}
				})
			}
		});
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

app.get('/auth/facebook',
	passport.authenticate(
		'facebook',
		{scope:'email'}
	)
)

app.get('/auth/facebook/callback',
	passport.authenticate(
		'facebook', 
		{
			successRedirect: '/welcome',
			failureRedirect: '/auth/login' 
		}
	)
);

app.post('/auth/register', function(req, res) {
	hasher({password: req.body.password}, function(err, pass, salt, hash) {
		var user = {
			authId: 'local:' + req.body.username,
			username: req.body.username,
			password: hash,
			salt: salt,
			displayName: req.body.displayName
		};
		var sql = "INSERT INTO users SET ?";
		conn.query(sql, user, function(err, results) {
			if(err) {
				console.log(err);
				res.status(500);
			} else {
				req.login(user, function(err) {
					req.session.save(function() {
						res.redirect('/welcome');
					})
				});
			}
		});

		// users.push(user);
		// req.login(user, function(err) {
		// 	req.session.save(function() {
		// 		res.redirect('/welcome');
		// 	})
		// });
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
		<a href="/auth/facebook">facebook</a>
	`;
	res.send(output);
})

app.listen(3002, function () {
	console.log('multi-user app listening on port 3002!')
})
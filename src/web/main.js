var realm = 'realm1';
var authid = 'controller';
var ticket = '64=t5tQPnGLh+PcK';


var ws = new Wampy('wss://127.0.0.1/ws', {
    realm: realm,
    authid: authid,
    authmethods: ['ticket'],
    onChallenge: function (method, info) {
        return ticket;
    },
    onConnect: function () {
        console.log('Connected.');
    },
    onError: function (details) { console.log(details); }
});



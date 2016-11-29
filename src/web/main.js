// Global terminal functions.
var term = $('body').terminal(
    function(command, term) {
        if (command !== '') {
            try {
                var result = window.eval(command);
                if (result !== undefined) {
                    term.echo(String(result));
                }
            } catch (e) {
                term.error(String(e));
            }
        } else {
            term.echo('');
        }
    }, {
        prompt: '> ',
        greetings:  '██████╗ ███████╗██╗  ██╗████████╗███████╗██████╗\n' +
                    '██╔══██╗██╔════╝╚██╗██╔╝╚══██╔══╝██╔════╝██╔══██╗\n' +
                    '██║  ██║█████╗   ╚███╔╝    ██║   █████╗  ██████╔╝\n' +
                    '██║  ██║██╔══╝   ██╔██╗    ██║   ██╔══╝  ██╔══██╗\n' +
                    '██████╔╝███████╗██╔╝ ██╗   ██║   ███████╗██║  ██║\n' +
                    '╚═════╝ ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═╝\n' +
                    '-------------------------------------------------\n'
    });

function echo(text) {
    var dt = moment().format('HH:mm:ss.SSS');
    term.echo('[[;#0e0;#000]&#91;' + dt + '&#93;] ' + text);
}

function echo2(dt, text) {
    dt = moment.unix(dt).format('HH:mm:ss.SSS');
    term.echo('[[;#ff7f00;#000]&#91;' + dt + '&#93;] ' + text);
}

// System messages.
echo('Starting system.');
echo('Public IP is ' + ip + '.');


// WAMP.
var realm = 'realm1';
var authid = 'controller';
var ticket = '64=t5tQPnGLh+PcK';

var ws = new Wampy('/ws', {
    realm: realm,
    authid: authid,
    onChallenge: function (method, info) {
        return ticket;
    },
    onConnect: function () {
        echo('Connected to WAMP server.');
    },
    onClose: function () {
        echo('Connection closed.')
    },
    onReconnectSuccess: function () {
        echo('Successfully reconnected.');
    },
    onError: function (details) {
        echo('WebSocket connection error.');
    }
});


// Logging.
ws.subscribe('arm.log', {
    onSuccess: function () {
        echo('Subscribed to device logging.');
    },
    onError: function (err, details) {
        echo('Unable to subscribe to device logging: ' + err + '.');
    },
    onEvent: function (arrayPayload, objectPayload) {
        var dt = arrayPayload[0];
        var msg = arrayPayload[1];
        echo2(dt, msg);
    }
});


// Voice.
var voice = {};
var synth = window.speechSynthesis;
var preferredVoice = 'Google US English';

voice.populate = function () {
    var voices = synth.getVoices();

    if (voices.length > 0 && voice.v === undefined) {
        echo('Populated voices.');

        var v = voices.filter(function (voice) {
            return voice.name === preferredVoice;
        })[0];

        if (v !== undefined) {
            voice.v = v;
        } else {
            voice.v = voices[0];
        }

        echo('Using voice "' + voice.v.name + '."');
    }
};

voice.speak = function (text) {
    echo('Speaking: "' + text + '"');

    var msg = new SpeechSynthesisUtterance();

    msg.text = text;
    msg.voice = voice.v;

    synth.speak(msg);
};

ws.register('controller.speak', {
    rpc: voice.speak,
    onSuccess: function (data) {
        echo('Successfully registered speaking function.');
    },
    onError: function (err, details) {
        echo('Unable ot register speaking funciton: ' + err + '.');
    }
});

voice.populate();
synth.onvoiceschanged = voice.populate;


// Recognition.
var control = {};

control.hello = function () {
    voice.speak('Hello. I am alive.');
};

control.call = function (f, payload) {
    if (payload === undefined) {
        payload = null;
    }

    ws.call('arm.' + f, payload, {
        onSuccess: function (data) {
            echo('Successfully called "' + f + '."')
        },
        onError: function (err, details, [arrayData, objectData]) {
            echo('Error while calling "' + f + '": ' + err + '.');
        }
    });
};

var commands = {
    'hello': {'regexp': /^\s*(?:hello|hi)?\s*dexter$/i, 'callback': control.hello},
    '(dexter) draw today\'s weather': function () {
        control.call('draw_weather');
    },
    '(dexter) ready': function () {
        control.call('ready');
    },
    '(dexter) stop': function () {
        control.call('stop');
    },
    '(dexter) info': function () {
        control.call('info');
    },
    '(dexter) zero': function () {
        control.call('zero');
    },
    '(dexter) move :direction :float': function (direction, float) {
        control.call('move_relative', [direction, float]);
    },
    '(dexter) move :x, :y, :z': function (x, y, z) {
        control.call('move_absolute', [x, y, z]);
    },
    '(dexter) set :parameter :float': function (parameter, float) {
        control.call('set_parameter', [parameter, float]);
    },
    '(dexter) save point :num': function (num) {
        control.call('save_point', [num]);
    }
};

annyang.addCommands(commands);
annyang.addCallback('start', function () {
    echo('Voice recognition initialized.');
    annyang.removeCallback('start');
});

annyang.debug();
annyang.start();



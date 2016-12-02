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
    if (text === undefined || text === null || text === '') {
        return;
    }

    var msg = new SpeechSynthesisUtterance();

    msg.text = text;
    msg.voice = voice.v;
    msg.onend = function (e) {
        annyang.resume();
    };
    msg.onstart = function (e) {
        annyang.pause();
    };

    synth.speak(msg);
};

ws.register('controller.speak', {
    rpc: voice.speak,
    onSuccess: function (data) {
        echo('Successfully registered speaking function.');
    },
    onError: function (err, details) {
        echo('Unable ot register speaking function: ' + err + '.');
    }
});

voice.populate();
synth.onvoiceschanged = voice.populate;


// Recognition.
var control = {};

control.call = function (f, payload, say_success, say_error) {
    if (payload === undefined) {
        payload = null;
    }

    echo('Calling "' + f + '" with payload [' + String(payload) + '].');

    ws.call('arm.' + f, payload, {
        onSuccess: function () {
            echo('Successfully called "' + f + '."');
            ws.call('controller.speak', say_success, function () { return undefined; });
        },
        onError: function (err) {
            echo('Error while calling "' + f + '": ' + err + '.');
            ws.call('controller.speak', say_error, function () { return undefined; });
        }
    });
};

control.activate = function () {
    annyang.removeCommands();
    annyang.addCommands(active_commands);
    echo('Voice control activated.');
    ws.call('controller.speak', 'Hello. I am alive.', function () { return undefined; });
};

control.deactivate = function () {
    annyang.removeCommands();
    annyang.addCommands(inactive_commands);
    echo('Voice control deactivated.');
    ws.call('controller.speak', 'Going to sleep.', function () { return undefined; });
};

var inactive_commands = {
    'dexter': function () {
        control.activate();
    }
};

var active_commands = {
    '(dexter) calibrate': function () {
        control.call('calibrate');
    },
    '(dexter) get position': function () {
        control.call('get_position');
    },
    '(dexter) home': function () {
        control.call('home');
    },
    '(dexter) info': function () {
        control.call('info');
    },
    '(dexter) ready': function () {
        control.call('ready', null, null, 'I can\'t feel my arm. Did you power it on?');
    },
    '(dexter) sleep': function () {
        control.deactivate();
    },
    '(dexter) stop': function () {
        control.call('stop');
    },
    '(dexter) draw the weather': function () {
        control.call('draw_weather');
    },
    '(dexter) draw the weather in :value :units': function (value, units) {
        control.call('draw_forecast', [value, units]);
    },
    '(dexter) draw *q': function (q) {
        control.call('draw_image', q);
    },
    '(dexter) draw index :i query *q': function (i, q) {
        control.call('draw_image', [q, i]);
    },
    '(dexter) trace *q': function (q) {
        control.call('trace_image', q);
    },
    '(dexter) trace index :i query *q': function (i, q) {
        control.call('trac_image', [i, q]);
    },
    '(dexter) move :direction :float': function (direction, float) {
        control.call('move_relative', [direction, float]);
    },
    '(dexter) move (to) :x, :y, :z': function (x, y, z) {
        control.call('move_absolute', [x, y, z]);
    },
    '(dexter) load point :name': function (name) {
        control.call('load_point', name);
    },
    '(dexter) save point as :name': function (name) {
        control.call('save_point', name);
    },
    '(dexter) set :parameter :float': function (parameter, float) {
        control.call('set_parameter', [parameter, float]);
    },
    '(dexter) *input': function (input) {
        control.call('chat', input);
    }
};


annyang.addCommands(inactive_commands);
annyang.addCallback('start', function () {
    echo('Voice recognition initialized.');
    annyang.removeCallback('start');
});

annyang.debug();
annyang.start();




//<![CDATA[

// a few things don't have var in front of them - they update already existing variables the game needs
lanesSide = 2;
patchesAhead = 4;
patchesBehind = 4;
trainIterations = 10000000;

var num_inputs = (lanesSide) * (patchesAhead + patchesBehind);
var num_actions = 5;
var temporal_window = 3;
var network_size = num_inputs * temporal_window;

var layer_defs = [];
layer_defs.push({
    type:'input',
    out_sx: (lanesSide),
    out_sy: (patchesAhead + patchesBehind),
    out_depth: temporal_window
});
layer_defs.push({type:'conv', sx:3, sy: 4, filters:3, stride:1, pad:1, activation:'relu'});
layer_defs.push({type:'conv', sx:2, sy: 4, filters:3, stride:1, pad:1, activation:'relu'});
layer_defs.push({type:'pool', sx:1, sy: 4, stride:1});
layer_defs.push({type: 'regression', num_neurons: num_actions});

var tdtrainer_options = {
    method: 'adadelta',
    learning_rate: 0.001,
    batch_size: 5,
    l2_decay:0.0001
};

var opt = {};
opt.temporal_window = temporal_window;
opt.experience_size = 30000;
opt.start_learn_threshold = 1000;
opt.gamma = 0.7;
opt.learning_steps_total = 200000;
opt.learning_steps_burnin = 3000;
opt.epsilon_min = 0.10;
opt.epsilon_test_time = 0.10;
opt.layer_defs = layer_defs;
opt.tdtrainer_options = tdtrainer_options;

brain = new deepqlearn.Brain(num_inputs, num_actions, opt);

learn = function (state, lastReward) {
    brain.backward(lastReward);
    var action = brain.forward(state);

    draw_net();
    draw_stats();

    return action;
}

//]]>


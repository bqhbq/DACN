/**
 * Draw Road Network
 */
id = Math.random().toString(36).substring(2, 15);

BACKGROUND_COLOR = 0xe8ebed;
LANE_COLOR = 0x586970;
LANE_BORDER_WIDTH = 1;
LANE_BORDER_COLOR = 0x82a8ba;
LANE_INNER_COLOR = 0xbed8e8;
LANE_DASH = 10;
LANE_GAP = 12;
TRAFFIC_LIGHT_WIDTH = 3;
MAX_TRAFFIC_LIGHT_NUM = 100000;
ROTATE = 90;

CAR_LENGTH = 5;
CAR_WIDTH = 2;
CAR_COLOR = 0xe8bed4;

CAR_COLORS = [0xf2bfd7, // pink
            0xb7ebe4,   // cyan
            0xdbebb7,   // blue
            0xf5ddb5, 
            0xd4b5f5];
CAR_COLORS_NUM = CAR_COLORS.length;

NUM_CAR_POOL = 150000;

LIGHT_RED = 0xdb635e;
LIGHT_GREEN = 0x85ee00;

TURN_SIGNAL_COLOR = 0xFFFFFF;
TURN_SIGNAL_WIDTH   = 1;
TURN_SIGNAL_LENGTH  = 5;

var simulation, roadnet, steps;
var nodes = {};
var edges = {};
var logs;
var gettingLog = false;

let Application = PIXI.Application,
    Sprite = PIXI.Sprite,
    Graphics = PIXI.Graphics,
    Container = PIXI.Container,
    ParticleContainer = PIXI.particles.ParticleContainer,
    Texture = PIXI.Texture,
    Rectangle = PIXI.Rectangle
;

var controls = new function () {
    this.replaySpeedMax = 1;
    this.replaySpeedMin = 0.01;
    this.replaySpeed = 0.5;
    this.paused = false;
};

var trafficLightsG = {};

var app, viewport, renderer, simulatorContainer, carContainer, trafficLightContainer;
var turnSignalContainer;
var carPool;

var cnt = 0;
var frameElapsed = 0;
var totalStep;

var nodeCarNum = document.getElementById("car-num");
var nodeProgressPercentage = document.getElementById("progress-percentage");
var nodeTotalStep = document.getElementById("total-step-num");
var nodeCurrentStep = document.getElementById("current-step-num");
var nodeSelectedEntity = document.getElementById("selected-entity");

var SPEED = 3, SCALE_SPEED = 1.01;
var LEFT = 37, UP = 38, RIGHT = 39, DOWN = 40;
var MINUS = 189, EQUAL = 187, P = 80;
var LEFT_BRACKET = 219, RIGHT_BRACKET = 221; 
var ONE = 49, TWO = 50;
var SPACE = 32;

var keyDown = new Set();

var turnSignalTextures = [];

let pauseButton = document.getElementById("pause");
let nodeCanvas = document.getElementById("simulator-canvas");
let replayControlDom = document.getElementById("replay-control");
let replaySpeedDom = document.getElementById("replay-speed");

let loading = false;
let infoDOM = document.getElementById("info");
let selectedDOM = document.getElementById("selected-entity");

function infoAppend(msg) {
    infoDOM.innerText += "- " + msg + "\n";
}

function infoReset() {
    infoDOM.innerText = "";
}

/**
 * Upload files
 */
let ready = false;

let roadnetData = [];
let replayData = [];
let chartData = [];
let flData = [];

function handleChooseFile(v, label_dom) {
    return function(evt) {
        let file = evt.target.files[0];
        label_dom.innerText = file.name;
    }
}

function uploadFile(v, file, callback) {
    let reader = new FileReader();
    reader.onloadstart = function () {
        infoAppend("Loading " + file.name);
    };
    reader.onerror = function() {
        infoAppend("Loading " + file.name + "failed");
    }
    reader.onload = function (e) {
        infoAppend(file.name + " loaded");
        v[0] = e.target.result;
        callback();
    };
    try {
        reader.readAsText(file);
    } catch (e) {
        infoAppend("Loading failed");
        console.error(e.message);
    }
}

let debugMode = false;
let chartLog;
let showChart = false;
let chartConainterDOM = document.getElementById("chart-container");
function start() {
    if (loading) return;
    loading = true;
    infoReset();
    uploadFile(roadnetData, RoadnetFileDom.files[0], function(){
    uploadFile(replayData, ReplayFileDom.files[0], function(){
        let after_update = function() {
            infoAppend("drawing roadnet");
            ready = false;
            document.getElementById("guide").classList.add("d-none");
            hideCanvas();
            try {
                simulation = JSON.parse(roadnetData[0]);
            } catch (e) {
                infoAppend("Parsing roadnet file failed");
                loading = false;
                return;
            }
            try {
                logs = replayData[0].split('\n');
                logs.pop();
            } catch (e) {
                infoAppend("Reading replay file failed");
                loading = false;
                return;
            }

            totalStep = logs.length;
            if (showChart) {
                chartConainterDOM.classList.remove("d-none");
                let chart_lines = chartData[0].split('\n');
                if (chart_lines.length == 0) {
                    infoAppend("Chart file is empty");
                    showChart = false;
                }
                chartLog = [];
                for (let i = 0 ; i < totalStep ; ++i) {
                    step_data = chart_lines[i + 1].split(/[ \t]+/);
                    chartLog.push([]);
                    for (let j = 0; j < step_data.length; ++j) {
                        chartLog[i].push(parseFloat(step_data[j]));
                    }
                }
                chart.init(chart_lines[0], chartLog[0].length, totalStep);
            }else {
                chartConainterDOM.classList.add("d-none");
            }

            fl.init(FlFileDom.files.length ? flData[0] : null, totalStep);

            controls.paused = false;
            cnt = 0;
            debugMode = document.getElementById("debug-mode").checked;
            setTimeout(function () {
                try {
                    drawRoadnet();
                } catch (e) {
                    infoAppend("Drawing roadnet failed");
                    console.error(e.message);
                    loading = false;
                    return;
                }
                ready = true;
                loading = false;
                infoAppend("Start replaying");
            }, 200);
        };


        // Optional files are loaded in sequence: chart, then FL log.
        let load_fl = function() {
            if (FlFileDom.files.length) {
                uploadFile(flData, FlFileDom.files[0], after_update);
            } else {
                after_update();
            }
        };

        if (ChartFileDom.value) {
            showChart = true;
            uploadFile(chartData, ChartFileDom.files[0], load_fl);
        } else {
            showChart = false;
            load_fl();
        }

    }); // replay callback
    }); // roadnet callback
}

let RoadnetFileDom = document.getElementById("roadnet-file");
let ReplayFileDom = document.getElementById("replay-file");
let ChartFileDom = document.getElementById("chart-file");
let FlFileDom = document.getElementById("fl-file");

RoadnetFileDom.addEventListener("change",
    handleChooseFile(roadnetData, document.getElementById("roadnet-label")), false);
ReplayFileDom.addEventListener("change",
    handleChooseFile(replayData, document.getElementById("replay-label")), false);
ChartFileDom.addEventListener("change",
    handleChooseFile(chartData, document.getElementById("chart-label")), false);
FlFileDom.addEventListener("change",
    handleChooseFile(flData, document.getElementById("fl-label")), false);

document.getElementById("start-btn").addEventListener("click", start);

document.getElementById("slow-btn").addEventListener("click", function() {
    updateReplaySpeed(controls.replaySpeed - 0.1);
})

document.getElementById("fast-btn").addEventListener("click", function() {
    updateReplaySpeed(controls.replaySpeed + 0.1);
})

function updateReplaySpeed(speed){
    speed = Math.min(speed, 1);
    speed = Math.max(speed, 0);
    controls.replaySpeed = speed;
    replayControlDom.value = speed * 100;
    replaySpeedDom.innerHTML = speed.toFixed(2);
}

updateReplaySpeed(0.5);

replayControlDom.addEventListener('change', function(e){
    updateReplaySpeed(replayControlDom.value / 100);
});

document.addEventListener('keydown', function(e) {
    if (e.keyCode == P) {
        controls.paused = !controls.paused;
    } else if (e.keyCode == ONE) {
        updateReplaySpeed(Math.max(controls.replaySpeed / 1.5, controls.replaySpeedMin));
    } else if (e.keyCode == TWO ) {
        updateReplaySpeed(Math.min(controls.replaySpeed * 1.5, controls.replaySpeedMax));
    } else if (e.keyCode == LEFT_BRACKET) {
        cnt = (cnt - 1) % totalStep;
        cnt = (cnt + totalStep) % totalStep;
        drawStep(cnt);
    } else if (e.keyCode == RIGHT_BRACKET) {
        cnt = (cnt + 1) % totalStep;
        drawStep(cnt);
    } else {
        keyDown.add(e.keyCode)
    }
});

document.addEventListener('keyup', (e) => keyDown.delete(e.keyCode));

nodeCanvas.addEventListener('dblclick', function(e){
    controls.paused = !controls.paused;
});

pauseButton.addEventListener('click', function(e){
    controls.paused = !controls.paused;
});

function initCanvas() {
    app = new Application({
        width: nodeCanvas.offsetWidth,
        height: nodeCanvas.offsetHeight,
        transparent: false,
        backgroundColor: BACKGROUND_COLOR
    });

    nodeCanvas.appendChild(app.view);
    app.view.classList.add("d-none");

    renderer = app.renderer;
    renderer.interactive = true;
    renderer.autoResize = true;

    renderer.resize(nodeCanvas.offsetWidth, nodeCanvas.offsetHeight);
    app.ticker.add(run);
}

function showCanvas() {
    document.getElementById("spinner").classList.add("d-none");
    app.view.classList.remove("d-none");
}

function hideCanvas() {
    document.getElementById("spinner").classList.remove("d-none");
    app.view.classList.add("d-none");
}

function drawRoadnet() {
    if (simulatorContainer) {
        simulatorContainer.destroy(true);
    }
    app.stage.removeChildren();
    viewport = new Viewport.Viewport({
        screenWidth: window.innerWidth,
        screenHeight: window.innerHeight,
        interaction: app.renderer.plugins.interaction
    });
    viewport
        .drag()
        .pinch()
        .wheel()
        .decelerate();
    app.stage.addChild(viewport);
    simulatorContainer = new Container();
    viewport.addChild(simulatorContainer);

    roadnet = simulation.static;
    nodes = [];
    edges = [];
    trafficLightsG = {};

    for (let i = 0, len = roadnet.nodes.length;i < len;++i) {
        node = roadnet.nodes[i];
        node.point = new Point(transCoord(node.point));
        nodes[node.id] = node;
    }

    for (let i = 0, len = roadnet.edges.length;i < len;++i) {
        edge = roadnet.edges[i];
        edge.from = nodes[edge.from];
        edge.to = nodes[edge.to];
        for (let j = 0, len = edge.points.length;j < len;++j) {
            edge.points[j] = new Point(transCoord(edge.points[j]));
        }
        edges[edge.id] = edge;
    }

    /**
     * Draw Map
     */
    trafficLightContainer = new ParticleContainer(MAX_TRAFFIC_LIGHT_NUM, {tint: true});
    let mapContainer, mapGraphics;
    if (debugMode) {
        mapContainer = new Container();
        simulatorContainer.addChild(mapContainer);
    }else {
        mapGraphics = new Graphics();
        simulatorContainer.addChild(mapGraphics);
    }

    for (nodeId in nodes) {
        if (!nodes[nodeId].virtual) {
            let nodeGraphics;
            if (debugMode) {
                nodeGraphics = new Graphics();
                mapContainer.addChild(nodeGraphics);
            } else {
                nodeGraphics = mapGraphics;
            }
            drawNode(nodes[nodeId], nodeGraphics);
        }
    }
    for (edgeId in edges) {
        let edgeGraphics;
        if (debugMode) {
            edgeGraphics = new Graphics();
            mapContainer.addChild(edgeGraphics);
        } else {
            edgeGraphics = mapGraphics;
        }
        drawEdge(edges[edgeId], edgeGraphics);
    }
    let bounds = simulatorContainer.getBounds();
    simulatorContainer.pivot.set(bounds.x + bounds.width / 2, bounds.y + bounds.height / 2);
    simulatorContainer.position.set(renderer.width / 2, renderer.height / 2);
    simulatorContainer.addChild(trafficLightContainer);

    /**
     * Settings for Cars
     */
    TURN_SIGNAL_LENGTH = CAR_LENGTH;
    TURN_SIGNAL_WIDTH  = CAR_WIDTH / 2;

    var carG = new Graphics();
    carG.lineStyle(0);
    carG.beginFill(0xFFFFFF, 0.8);
    carG.drawRect(0, 0, CAR_LENGTH, CAR_WIDTH);

    let carTexture = renderer.generateTexture(carG);

    let signalG = new Graphics();
    signalG.beginFill(TURN_SIGNAL_COLOR, 0.7).drawRect(0,0, TURN_SIGNAL_LENGTH, TURN_SIGNAL_WIDTH)
           .drawRect(0, 3 * CAR_WIDTH - TURN_SIGNAL_WIDTH, TURN_SIGNAL_LENGTH, TURN_SIGNAL_WIDTH).endFill();
    let turnSignalTexture = renderer.generateTexture(signalG);

    let signalLeft = new Texture(turnSignalTexture, new Rectangle(0, 0, TURN_SIGNAL_LENGTH, CAR_WIDTH));
    let signalStraight = new Texture(turnSignalTexture, new Rectangle(0, CAR_WIDTH, TURN_SIGNAL_LENGTH, CAR_WIDTH));
    let signalRight = new Texture(turnSignalTexture, new Rectangle(0, CAR_WIDTH * 2, TURN_SIGNAL_LENGTH, CAR_WIDTH));
    turnSignalTextures = [signalLeft, signalStraight, signalRight];


    carPool = [];
    if (debugMode)
        carContainer = new Container();
    else
        carContainer = new ParticleContainer(NUM_CAR_POOL, {rotation: true, tint: true});


    turnSignalContainer = new ParticleContainer(NUM_CAR_POOL, {rotation: true, tint: true});
    simulatorContainer.addChild(carContainer);
    simulatorContainer.addChild(turnSignalContainer);
    for (let i = 0, len = NUM_CAR_POOL;i < len;++i) {
        //var car = Sprite.fromImage("images/car.png")
        let car = new Sprite(carTexture);
        let signal = new Sprite(turnSignalTextures[1]);
        car.anchor.set(1, 0.5);

        if (debugMode) {
            car.interactive = true;
            car.on('mouseover', function () {
                selectedDOM.innerText = car.name;
                car.alpha = 0.8;
            });
            car.on('mouseout', function () {
                // selectedDOM.innerText = "";
                car.alpha = 1;
            });
        }
        signal.anchor.set(1, 0.5);
        carPool.push([car, signal]);
    }
    fl.drawOverlay();
    showCanvas();

    return true;
}

function appendText(id, text) {
    let p = document.createElement("span");
    p.innerText = text;
    document.getElementById("info").appendChild(p);
    document.getElementById("info").appendChild(document.createElement("br"));
}

var statsFile = "";
var withRange = false;
var nodeStats, nodeRange;

initCanvas();


function transCoord(point) {
    return [point[0], -point[1]];
}

PIXI.Graphics.prototype.drawLine = function(pointA, pointB) {
    this.moveTo(pointA.x, pointA.y);
    this.lineTo(pointB.x, pointB.y);
}

PIXI.Graphics.prototype.drawDashLine = function(pointA, pointB, dash = 16, gap = 8) {
    let direct = pointA.directTo(pointB);
    let distance = pointA.distanceTo(pointB);

    let currentPoint = pointA;
    let currentDistance = 0;
    let length;
    let finish = false;
    while (true) {
        this.moveTo(currentPoint.x, currentPoint.y);
        if (currentDistance + dash >= distance) {
            length = distance - currentDistance;
            finish = true;
        } else {
            length = dash
        }
        currentPoint = currentPoint.moveAlong(direct, length);
        this.lineTo(currentPoint.x, currentPoint.y);
        if (finish) break;
        currentDistance += length;

        if (currentDistance + gap >= distance) {
            break;
        } else {
            currentPoint = currentPoint.moveAlong(direct, gap);
            currentDistance += gap;
        }
    }
};

function drawNode(node, graphics) {
    graphics.beginFill(LANE_COLOR);
    let outline = node.outline;
    for (let i = 0 ; i < outline.length ; i+=2) {
        outline[i+1] = -outline[i+1];
        if (i == 0)
            graphics.moveTo(outline[i], outline[i+1]);
        else
            graphics.lineTo(outline[i], outline[i+1]);
    }
    graphics.endFill();

    if (debugMode) {
        graphics.hitArea = new PIXI.Polygon(outline);
        graphics.interactive = true;
        graphics.on("mouseover", function () {
            selectedDOM.innerText = node.id;
            graphics.alpha = 0.5;
        });
        graphics.on("mouseout", function () {
            graphics.alpha = 1;
        });
    }

}

function drawEdge(edge, graphics) {
    let from = edge.from;
    let to = edge.to;
    let points = edge.points;

    let pointA, pointAOffset, pointB, pointBOffset;
    let prevPointBOffset = null;

    let roadWidth = 0;
    edge.laneWidths.forEach(function(l){
        roadWidth += l;
    }, 0);

    let coords = [], coords1 = [];

    for (let i = 1;i < points.length;++i) {
        if (i == 1){
            pointA = points[0].moveAlongDirectTo(points[1], from.virtual ? 0 : from.width);
            pointAOffset = points[0].directTo(points[1]).rotate(ROTATE);
        } else {
            pointA = points[i-1];
            pointAOffset = prevPointBOffset;
        }
        if (i == points.length - 1) {
            pointB = points[i].moveAlongDirectTo(points[i-1], to.virtual ? 0 : to.width);
            pointBOffset = points[i-1].directTo(points[i]).rotate(ROTATE);
        } else {
            pointB = points[i];
            pointBOffset = points[i-1].directTo(points[i+1]).rotate(ROTATE);
        }
        prevPointBOffset = pointBOffset;

        lightG = new Graphics();
        lightG.lineStyle(TRAFFIC_LIGHT_WIDTH, 0xFFFFFF);
        lightG.drawLine(new Point(0, 0), new Point(1, 0));
        lightTexture = renderer.generateTexture(lightG);

        // Draw Traffic Lights
        if (i == points.length-1 && !to.virtual) {
            edgeTrafficLights = [];
            prevOffset = offset = 0;
            for (lane = 0;lane < edge.nLane;++lane) {
                offset += edge.laneWidths[lane];
                var light = new Sprite(lightTexture);
                light.anchor.set(0, 0.5);
                light.scale.set(offset - prevOffset, 1);
                point_ = pointB.moveAlong(pointBOffset, prevOffset);
                light.position.set(point_.x, point_.y);
                light.rotation = pointBOffset.getAngleInRadians();
                edgeTrafficLights.push(light);
                prevOffset = offset;
                trafficLightContainer.addChild(light);
            }
            trafficLightsG[edge.id] = edgeTrafficLights;
        }

        // Draw Roads
        graphics.lineStyle(LANE_BORDER_WIDTH, LANE_BORDER_COLOR, 1);
        graphics.drawLine(pointA, pointB);

        pointA1 = pointA.moveAlong(pointAOffset, roadWidth);
        pointB1 = pointB.moveAlong(pointBOffset, roadWidth);

        graphics.lineStyle(0);
        graphics.beginFill(LANE_COLOR);

        coords = coords.concat([pointA.x, pointA.y, pointB.x, pointB.y]);
        coords1 = coords1.concat([pointA1.y, pointA1.x, pointB1.y, pointB1.x]);

        graphics.drawPolygon([pointA.x, pointA.y, pointB.x, pointB.y, pointB1.x, pointB1.y, pointA1.x, pointA1.y]);
        graphics.endFill();

        offset = 0;
        for (let lane = 0, len = edge.nLane-1;lane < len;++lane) {
            offset += edge.laneWidths[lane];
            graphics.lineStyle(LANE_BORDER_WIDTH, LANE_INNER_COLOR);
            graphics.drawDashLine(pointA.moveAlong(pointAOffset, offset), pointB.moveAlong(pointBOffset, offset), LANE_DASH, LANE_GAP);
        }

        offset += edge.laneWidths[edge.nLane-1];

        // graphics.lineStyle(LANE_BORDER_WIDTH, LANE_BORDER_COLOR);
        // graphics.drawLine(pointA.moveAlong(pointAOffset, offset), pointB.moveAlong(pointBOffset, offset));
    }

    if (debugMode) {
        coords = coords.concat(coords1.reverse());
        graphics.interactive = true;
        graphics.hitArea = new PIXI.Polygon(coords);
        graphics.on("mouseover", function () {
            graphics.alpha = 0.5;
            selectedDOM.innerText = edge.id;
        });

        graphics.on("mouseout", function () {
            graphics.alpha = 1;
        });
    }
}

function run(delta) {
    let redraw = false;

    if (ready && (!controls.paused || redraw)) {
        try {
            drawStep(cnt);
        }catch (e) {
            infoAppend("Error occurred when drawing");
            ready = false;
        }
        if (!controls.paused) {
            frameElapsed += 1;
            if (frameElapsed >= 1 / controls.replaySpeed ** 2) {
                cnt += 1;
                frameElapsed = 0;
                if (cnt == totalStep) cnt = 0;
            }
        }
    }
}

function _statusToColor(status) {
    switch (status) {
        case 'r':
            return LIGHT_RED;
        case 'g':
            return LIGHT_GREEN;
        default:
            return 0x808080;  
    }
}

function stringHash(str) {
    let hash = 0;
    let p = 127, p_pow = 1;
    let m = 1e9 + 9;
    for (let i = 0; i < str.length; i++) {
        hash = (hash + str.charCodeAt(i) * p_pow) % m;
        p_pow = (p_pow * p) % m;
    }
    return hash;
}

function drawStep(step) {
    if (showChart && (step > chart.ptr || step == 0)) {
        if (step == 0) {
            chart.clear();
        }
        chart.ptr = step;
        chart.addData(chartLog[step]);
    }

    let [carLogs, tlLogs] = logs[step].split(';');

    tlLogs = tlLogs.split(',');
    carLogs = carLogs.split(',');
    
    let tlLog, tlEdge, tlStatus;
    for (let i = 0, len = tlLogs.length;i < len;++i) {
        tlLog = tlLogs[i].split(' ');
        tlEdge = tlLog[0];
        tlStatus = tlLog.slice(1);
        for (let j = 0, len = tlStatus.length;j < len;++j) {
            trafficLightsG[tlEdge][j].tint = _statusToColor(tlStatus[j]);
            if (tlStatus[j] == 'i' ) {
                trafficLightsG[tlEdge][j].alpha = 0;
            }else{
                trafficLightsG[tlEdge][j].alpha = 1;
            }
        }
    }

    carContainer.removeChildren();
    turnSignalContainer.removeChildren();
    let carLog, position, length, width;
    for (let i = 0, len = carLogs.length - 1;i < len;++i) {
        carLog = carLogs[i].split(' ');
        position = transCoord([parseFloat(carLog[0]), parseFloat(carLog[1])]);
        length = parseFloat(carLog[5]);
        width = parseFloat(carLog[6]);
        carPool[i][0].position.set(position[0], position[1]);
        carPool[i][0].rotation = 2*Math.PI - parseFloat(carLog[2]);
        carPool[i][0].name = carLog[3];
        let carColorId = stringHash(carLog[3]) % CAR_COLORS_NUM;
        carPool[i][0].tint = CAR_COLORS[carColorId];
        carPool[i][0].width = length;
        carPool[i][0].height = width;
        carContainer.addChild(carPool[i][0]);

        let laneChange = parseInt(carLog[4]) + 1;
        carPool[i][1].position.set(position[0], position[1]);
        carPool[i][1].rotation = carPool[i][0].rotation;
        carPool[i][1].texture = turnSignalTextures[laneChange];
        carPool[i][1].width = length;
        carPool[i][1].height = width;
        turnSignalContainer.addChild(carPool[i][1]);
    }
    fl.update(step);
    nodeCarNum.innerText = carLogs.length-1;
    nodeTotalStep.innerText = totalStep;
    nodeCurrentStep.innerText = cnt+1;
    nodeProgressPercentage.innerText = (cnt / totalStep * 100).toFixed(2) + "%";
    if (statsFile != "") {
        if (withRange) nodeRange.value = stats[step][1];
        nodeStats.innerText = stats[step][0].toFixed(2);
    }
}

/*
Chart
 */
let chart = {
    max_steps: 3600,
    data: {
        labels: [],
        series: [[]]
    },
    options: {
        showPoint: false,
        lineSmooth: false,
        axisX: {
            showGrid: false,
            showLabel: false
        }
    },
    init : function(title, series_cnt, max_step){
        document.getElementById("chart-title").innerText = title;
        this.max_steps = max_step;
        this.data.labels = new Array(this.max_steps);
        this.data.series = [];
        for (let i = 0 ; i < series_cnt ; ++i)
            this.data.series.push([]);
        this.chart = new Chartist.Line('#chart', this.data, this.options);
    },
    addData: function (value) {
        for (let i = 0 ; i < value.length; ++i) {
            this.data.series[i].push(value[i]);
            if (this.data.series[i].length > this.max_steps) {
                this.data.series[i].shift();
            }
        }
        this.chart.update();
    },
    clear: function() {
        for (let i = 0 ; i < this.data.series.length ; ++i)
            this.data.series[i] = [];
    },
    ptr: 0
};


/*
Federated Learning view (optional "FL Log File" = fl_log.json written by main_demo_fl.py)

 - Side panel (#fl-panel): per-node local training stats, and the numbers of the
   latest FL round (n_k, FedAvg coefficients, norms, a worked example).
 - Map overlay: a "FL Server" box linked to every node agent. Right after each
   FL round the links animate: node -> server (upload backbone), aggregation at
   the server, server -> node (broadcast global backbone).
 */
FL_ANIM_STEPS = 24;          // replay steps the weight-exchange animation lasts
FL_COLOR_IDLE = 0x9aa5ad;
FL_COLOR_UP = 0xf08c00;      // uplink: node -> server
FL_COLOR_AGG = 0x7048e8;     // aggregation at the server
FL_COLOR_DOWN = 0x2f9e44;    // downlink: server -> node
FL_SERVER_OFFSET = 170;      // distance of the server box above the road network
FL_STAGES = ["idle", "up", "agg", "down"];

let fl = {
    enabled: false,
    data: null,
    decisionAt: [],   // step -> index of the latest decision at or before it (-1: none)
    roundAt: [],      // step -> index of the latest FL round at or before it (-1: none)
    panelDOM: document.getElementById("fl-panel"),
    linkG: null,
    serverG: null,
    serverText: null,
    serverPos: null,
    nodePos: {},
    lastKey: "",

    init: function(text, totalStep) {
        this.enabled = false;
        this.data = null;
        this.lastKey = "";
        this.panelDOM.classList.add("d-none");
        if (!text) return;
        try {
            this.data = JSON.parse(text);
        } catch (e) {
            infoAppend("Parsing FL log file failed");
            return;
        }
        this.decisionAt = this._indexBySteps(this.data.decisions, totalStep);
        this.roundAt = this._indexBySteps(this.data.rounds, totalStep);
        this.enabled = true;
        this.panelDOM.classList.remove("d-none");
        infoAppend("FL log loaded: " + this.data.rounds.length + " FL rounds");
    },

    // For every replay step, the index of the latest event whose step <= it.
    _indexBySteps: function(events, totalStep) {
        let result = new Array(totalStep);
        let ptr = -1;
        for (let step = 0; step < totalStep; ++step) {
            while (ptr + 1 < events.length && events[ptr + 1].step <= step) ptr += 1;
            result[step] = ptr;
        }
        return result;
    },

    drawOverlay: function() {
        if (!this.enabled) return;
        try {
            this.nodePos = {};
            let sumX = 0, minY = Infinity, count = 0;
            for (let nodeId in this.data.nodes) {
                if (!nodes[nodeId]) continue;
                let point = nodes[nodeId].point;
                this.nodePos[nodeId] = point;
                sumX += point.x;
                minY = Math.min(minY, point.y);
                count += 1;
            }
            if (count == 0) {
                infoAppend("FL log does not match this roadnet");
                this.enabled = false;
                this.panelDOM.classList.add("d-none");
                return;
            }
            // Canvas y grows downwards, so "above the map" means a smaller y.
            this.serverPos = new Point(sumX / count, minY - FL_SERVER_OFFSET);

            this.linkG = new Graphics();
            this.serverG = new Graphics();
            simulatorContainer.addChild(this.linkG);
            simulatorContainer.addChild(this.serverG);

            this.serverText = new PIXI.Text("FL Server", {
                fontFamily: "Arial", fontSize: 16, fontWeight: "bold", fill: 0xffffff, align: "center"
            });
            this.serverText.anchor.set(0.5, 0.5);
            this.serverText.position.set(this.serverPos.x, this.serverPos.y);
            simulatorContainer.addChild(this.serverText);

            for (let nodeId in this.nodePos) {
                let label = new PIXI.Text(this.data.nodes[nodeId], {
                    fontFamily: "Arial", fontSize: 14, fontWeight: "bold", fill: 0x343a40
                });
                label.anchor.set(0, 1);
                label.position.set(this.nodePos[nodeId].x + 22, this.nodePos[nodeId].y - 22);
                simulatorContainer.addChild(label);
            }
        } catch (e) {
            console.error(e.message);
            this.enabled = false;
        }
    },

    // Which part of the weight exchange is being shown at this step, and how far along it is.
    _stage: function(step, round) {
        if (!round) return {name: "idle", progress: 0};
        let elapsed = step - round.step;
        if (elapsed < 0 || elapsed >= FL_ANIM_STEPS) return {name: "idle", progress: 0};
        let third = FL_ANIM_STEPS / 3;
        let index = Math.min(2, Math.floor(elapsed / third));
        return {name: FL_STAGES[index + 1], progress: (elapsed - index * third) / third};
    },

    update: function(step) {
        if (!this.enabled) return;
        try {
            let decisionIdx = this.decisionAt[step];
            let roundIdx = this.roundAt[step];
            let round = roundIdx >= 0 ? this.data.rounds[roundIdx] : null;
            let stage = this._stage(step, round);

            this._drawLinks(stage, round);

            // The panel only changes when a new decision/round/stage starts.
            let key = decisionIdx + "|" + roundIdx + "|" + stage.name;
            if (key != this.lastKey) {
                this.lastKey = key;
                let decision = decisionIdx >= 0 ? this.data.decisions[decisionIdx] : null;
                this.panelDOM.innerHTML = this._panelHTML(decision, round, stage);
            }
        } catch (e) {
            // Never let the FL view break the replay itself.
            console.error(e.message);
            this.enabled = false;
        }
    },

    _drawLinks: function(stage, round) {
        let color = {idle: FL_COLOR_IDLE, up: FL_COLOR_UP, agg: FL_COLOR_AGG, down: FL_COLOR_DOWN}[stage.name];
        let active = stage.name == "up" || stage.name == "down";

        this.linkG.clear();
        for (let nodeId in this.nodePos) {
            let nodePoint = this.nodePos[nodeId];
            this.linkG.lineStyle(active ? 3 : 1, active ? color : FL_COLOR_IDLE, active ? 0.9 : 0.5);
            this.linkG.drawDashLine(nodePoint, this.serverPos, 10, 6);
            if (active) {
                // A "weights packet" travelling along the link.
                let from = stage.name == "up" ? nodePoint : this.serverPos;
                let to = stage.name == "up" ? this.serverPos : nodePoint;
                let x = from.x + (to.x - from.x) * stage.progress;
                let y = from.y + (to.y - from.y) * stage.progress;
                this.linkG.lineStyle(0);
                this.linkG.beginFill(color);
                this.linkG.drawCircle(x, y, 7);
                this.linkG.endFill();
            }
        }

        this.serverG.clear();
        this.serverG.lineStyle(2, 0xffffff, 0.9);
        this.serverG.beginFill(stage.name == "idle" ? 0x495057 : color);
        this.serverG.drawRoundedRect(this.serverPos.x - 70, this.serverPos.y - 24, 140, 48, 8);
        this.serverG.endFill();
        this.serverText.text = "FL Server\n" + (round ? "round " + round.fl_round : "waiting");
    },

    _esc: function(value) {
        return String(value).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    },

    _num: function(value, digits) {
        return (value === null || value === undefined) ? "–" : Number(value).toFixed(digits);
    },

    _panelHTML: function(decision, round, stage) {
        let esc = this._esc, num = this._num;
        let html = "<h6>Federated Learning</h6>";
        html += "<div class='fl-muted'>Local trainer: <b>" + esc(this.data.trainer) + "</b> · FL round every "
            + esc(this.data.fl_interval) + " steps · rounds done: <b>" + (round ? round.fl_round : 0) + "</b></div>";

        let stageText = {
            idle: "Local training at each node (no weights are being sent)",
            up: "① UPLINK: Node Agents ──▶ FL Server (backbone only)",
            agg: "② AGGREGATE at FL Server: w* = Σ (n_k/N)·w_k",
            down: "③ DOWNLINK: FL Server ──▶ Node Agents (broadcast w*)"
        }[stage.name];
        let stageClass = {idle: "", up: "fl-up", agg: "fl-agg", down: "fl-down"}[stage.name];
        html += "<div class='fl-stage " + stageClass + "'>" + stageText + "</div>";

        // Local training at each node
        html += "<table><thead><tr><th>Node</th><th>phase</th><th>action</th><th>reward</th><th>loss</th><th>n_k</th></tr></thead><tbody>";
        for (let nodeId in this.data.nodes) {
            let info = decision ? decision.nodes[nodeId] : null;
            html += "<tr><td><b>" + esc(this.data.nodes[nodeId]) + "</b></td>";
            if (info) {
                let action = (info.action === null || info.action === undefined) ? "–" : info.action;
                html += "<td>" + esc(info.phase) + "</td><td>" + esc(action) + " <span class='fl-muted'>(" + esc(info.result) + ")</span></td>"
                    + "<td>" + num(info.reward, 2) + "</td><td>" + num(info.loss, 4) + "</td><td>" + esc(info.n_k) + "</td>";
            } else {
                html += "<td colspan='5' class='fl-muted'>no decision yet</td>";
            }
            html += "</tr>";
        }
        html += "</tbody></table>";
        if (decision) {
            html += "<div class='fl-muted'>Backbone divergence between nodes (max |Δw|): <b>"
                + num(decision.divergence, 6) + "</b> at step " + decision.step + "</div>";
        }

        // Latest FL round
        if (!round) {
            html += "<div class='fl-muted mt-2'>No FL round yet.</div>";
            return html;
        }
        html += "<h6 class='mt-2'>FL round " + round.fl_round + " <span class='fl-muted'>(step " + (round.step + 1)
            + ", " + esc(round.algorithm) + ", " + round.param_tensors + " tensors)</span></h6>";
        html += "<table><thead><tr><th>Node</th><th>n_k</th><th>n_k/N</th><th>‖w_k‖ sent</th><th>‖w*−w_k‖</th><th>head</th></tr></thead><tbody>";
        for (let i = 0; i < round.clients.length; ++i) {
            let c = round.clients[i];
            html += "<tr><td><b>" + esc(c.name) + "</b></td><td>" + esc(c.n_k) + "</td><td>" + num(c.coefficient, 3)
                + "</td><td>" + num(c.upload_norm, 4) + "</td><td>" + num(c.delta_norm, 4) + "</td><td>"
                + (c.head_unchanged ? "kept ✓" : "CHANGED ✗") + "</td></tr>";
        }
        html += "</tbody></table>";

        let terms = [];
        for (let i = 0; i < round.clients.length; ++i) {
            terms.push(num(round.clients[i].coefficient, 3) + "×(" + num(round.example.client_values[i], 5) + ")");
        }
        html += "<div class='fl-formula'>" + esc(round.example.key) + "[" + round.example.index + "]: "
            + terms.join(" + ") + " = <b>" + num(round.example.global_value, 5) + "</b></div>";
        html += "<div class='fl-muted'>Each upload: " + round.clients[0].num_params.toLocaleString() + " params ("
            + num(round.clients[0].size_kb, 1) + " KB) · ‖w*‖ = " + num(round.global_norm, 4) + "</div>";
        html += "<div>Divergence: <b>" + num(round.divergence_before, 6) + "</b> → <b>"
            + num(round.divergence_after, 6) + "</b></div>";
        return html;
    }
};

from pathlib import Path
import json, html, os

class GridWorldHTMLRenderer:
    def __init__(self,  output_dir: str | os.PathLike = "renderings", filename='gridworld.html', cell_size=72,
                 show_state_ids=True, show_transitions=True, autoplay=False):
        self.output_dir = Path(output_dir)
        self.filename = str(filename)
        self.cell_size = int(cell_size)
        self.show_state_ids = bool(show_state_ids)
        self.show_transitions = bool(show_transitions)
        self.autoplay = bool(autoplay)
        self.frames = []
        self.env = None
        self.started = False

    def start(self, env):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.env = env
        self.frames = []
        self.started = True

        name = getattr(env, "name", None)
        if name is None:
            name = getattr(env, "name", "GridWorldMDP")

        if self.filename is None:
            safe_name = "".join(
                c if c.isalnum() or c in "-_." else "_"
                for c in str(name)
            )
            self.output_path = self.output_dir / f"{safe_name}.html"
        else:
            self.output_path = self.output_dir / self.filename
            if self.output_path.suffix.lower() != ".html":
                self.output_path = self.output_path.with_suffix(".html")

    def render(self, env, last):
        if not self.started:
            self.start(env)
        state, action, reward = last

        frame = {
            "state": None if state is None else int(state),
            "action": None if action is None else int(action),
            "reward": float(reward) if reward is not None else 0.0,
            "step": len(self.frames),
        }
        self.frames.append(frame)

    def stop(self, env):
        self.env = env or self.env
        if self.env is None:
            return


        self.output_dir.mkdir(parents=True, exist_ok=True)
        document = self._html()
        assert self.output_path is not None
        self.output_path.write_text(document, encoding="utf-8")

        self.started = False

    def _cells(self):
        e = self.env
        mapping = getattr(e, 'mapping', None)
        if mapping is None:
            return [[s // e.sizeY, s % e.sizeY] for s in range(e.nS)]
        return [list(e.from_s(int(mapping[s]))) for s in range(e.nS)]

    def _model(self):
        e = self.env
        cells = self._cells()
        maze = [[float(e.maze[x][y]) for y in range(e.sizeY)] for x in range(e.sizeX)]
        P = {}
        for s in range(e.nS):
            P[str(s)] = {}
            for a in range(e.nA):
                P[str(s)][str(a)] = [{'p': float(q[0]), 'next': int(q[1]),
                                      'done': bool(q[2]) if len(q) > 2 else False}
                                     for q in e.P[s][a]]
        R = {}
        for s in range(e.nS):
            R[str(s)] = {}
            for a in range(e.nA):
                try:
                    R[str(s)][str(a)] = float(e.R[s][a].mean())
                except Exception:
                    R[str(s)][str(a)] = 0.0
        return {'name': str(getattr(e,'displayname',None) or getattr(e,'name',None) or e.__class__.__name__),
                'sizeX': int(e.sizeX), 'sizeY': int(e.sizeY), 'nS': int(e.nS), 'nA': int(e.nA),
                'actions': [str(x) for x in getattr(e,'nameActions',[])], 'maze': maze,
                'goals': [int(s) for s in getattr(e,'goalstates',[])],
                'isd': [float(x) for x in getattr(e,'isd',[])], 'state_cells': cells,
                'transitions': P, 'rewards': R}

    def _html(self):
        m = self._model()
        data = json.dumps(m, ensure_ascii=False, separators=(',',':'))
        frames = json.dumps(self.frames, ensure_ascii=False, separators=(',',':'))
        title = html.escape(m['name'])
        cell = self.cell_size
        return f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#f4f6f8;color:#20242a;font-family:system-ui,sans-serif}}header{{height:72px;background:#fff;border-bottom:1px solid #d9dee5;padding:14px 20px;display:flex;justify-content:space-between}}h1{{margin:0;font-size:20px}}.sub{{color:#69717d;font-size:12px}}.layout{{display:grid;grid-template-columns:minmax(0,1fr) 330px;height:calc(100vh - 72px)}}.map{{overflow:hidden;background:#e9edf2}}#map{{width:100%;height:100%;display:block;touch-action:none;cursor:grab}}#map.dragging{{cursor:grabbing}}aside{{background:#fff;padding:15px;overflow:auto}}section{{border-bottom:1px solid #e1e5ea;padding-bottom:14px;margin-bottom:14px}}h2{{font-size:12px;text-transform:uppercase;color:#69717d;letter-spacing:.06em}}button,select{{font:inherit;border:1px solid #d9dee5;background:#fff;border-radius:7px;padding:7px;cursor:pointer}}button:hover{{background:#f1f4f8}}button.primary{{background:#3867d6;color:#fff}}.controls{{display:grid;grid-template-columns:repeat(5,1fr);gap:5px}}input[type=range]{{width:100%}}.row{{display:flex;justify-content:space-between;align-items:center;gap:8px}}.info{{background:#f7f8fa;border:1px solid #d9dee5;border-radius:8px;padding:9px;font-size:12px;line-height:1.45}}.check{{display:block;margin:8px 0;font-size:13px}}table{{width:100%;font-size:11px;border-collapse:collapse}}td,th{{padding:4px;border-bottom:1px solid #edf0f3;text-align:left}}.cell{{stroke:#cfd5dc;stroke-width:1;cursor:pointer}}.grid-cell:hover{{stroke:#3867d6;stroke-width:2}}.state-label,.goal-mark,.action-label{{pointer-events:none;text-anchor:middle}}.state-label{{font-size:11px;fill:#5b6470;dominant-baseline:central}}.goal-mark{{font-size:18px;font-weight:700;fill:#6b5200;dominant-baseline:central}}.current{{fill:#e85d04;stroke:#fff;stroke-width:3;pointer-events:none}}.initial-dot{{fill:#3867d6;opacity:.25;pointer-events:none}}.arrow{{fill:none;stroke:#3867d6;marker-end:url(#ah)}}.action-label{{font-size:11px;font-weight:700;fill:#20242a}}
@media(max-width:900px){{.layout{{grid-template-columns:1fr;height:auto}}.map{{height:65vh}}}}
</style></head><body><header><div><h1>{title}</h1><div class="sub">Interactive GridWorld renderer</div></div><div class="sub">{m['sizeX']} × {m['sizeY']} · {m['nS']} states · {m['nA']} actions</div></header>
<div class="layout"><div class="map"><svg id="map"><defs><marker id="ah" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7z"/></marker></defs><g id="vp"><g id="cells"></g><g id="initial"></g><g id="trans"></g><g id="labels"></g><g id="current"></g></g></svg></div>
<aside><section><h2>Playback</h2><div class="controls"><button id="first">|&lt;</button><button id="prev">&lt;</button><button id="play" class="primary">▶</button><button id="next">&gt;</button><button id="last">&gt;|</button></div><input id="timeline" type="range" min="0" max="0" value="0"><div class="row"><span id="fl">Frame 0 / 0</span><select id="speed"><option value="250">4×</option><option value="500" selected>2×</option><option value="1000">1×</option><option value="1500">0.67×</option><option value="2500">0.4×</option></select></div></section>
<section><h2>View</h2><div class="controls" style="grid-template-columns:repeat(3,1fr)"><button id="zo">−</button><button id="reset">Reset</button><button id="zi">+</button></div><label class="check"><input id="tr" type="checkbox"> Show transitions</label><label class="check"><input id="ids" type="checkbox"> Show state IDs</label><label class="check"><input id="ini" type="checkbox"> Show initial distribution</label></section>
<section><h2>Current step</h2><div id="step" class="info">No frame selected.</div></section><section><h2>Selected cell</h2><div id="cellinfo" class="info">Click a cell on the map.</div></section></aside></div>
<script>
const M={data}, F={frames}, C={cell}; const svg=document.getElementById('map'),vp=document.getElementById('vp');let fr=0,playing=false,timer=null,zoom=1,px=0,py=0;
const $=id=>document.getElementById(id); $('tr').checked={str(self.show_transitions).lower()};$('ids').checked={str(self.show_state_ids).lower()};
function cell(s){{return M.state_cells[s]}} function sid(x,y){{for(let s=0;s<M.nS;s++){{let c=cell(s);if(c[0]===x&&c[1]===y)return s}}return null}} function center(x,y){{return{{x:(y+.5)*C,y:(x+.5)*C}}}}
function draw(){{$('cells').innerHTML='';$('labels').innerHTML='';$('initial').innerHTML='';for(let x=0;x<M.sizeX;x++)for(let y=0;y<M.sizeY;y++){{let r=document.createElementNS('http://www.w3.org/2000/svg','rect');r.setAttribute('x',y*C);r.setAttribute('y',x*C);r.setAttribute('width',C);r.setAttribute('height',C);r.setAttribute('class','cell grid-cell');let v=M.maze[x][y];r.setAttribute('fill',v<=0?'#252a31':v>=2?'#f4c542':'#f8fafc');r.onclick=e=>{{e.stopPropagation();let s=sid(x,y);if(s!==null)select(s)}};$('cells').appendChild(r);let s=sid(x,y);if(s!==null&&$('ids').checked){{let c=center(x,y),t=document.createElementNS('http://www.w3.org/2000/svg','text');t.setAttribute('x',c.x);t.setAttribute('y',c.y);t.setAttribute('class','state-label');t.textContent=s;$('labels').appendChild(t)}}if(v>=2){{let c=center(x,y),t=document.createElementNS('http://www.w3.org/2000/svg','text');t.setAttribute('x',c.x);t.setAttribute('y',c.y);t.setAttribute('class','goal-mark');t.textContent='★';$('labels').appendChild(t)}}}}if($('ini').checked)for(let s=0;s<M.nS;s++){{let p=+(M.isd[s]||0);if(p>0){{let c=center(...cell(s)),d=document.createElementNS('http://www.w3.org/2000/svg','circle');d.setAttribute('cx',c.x);d.setAttribute('cy',c.y);d.setAttribute('r',8+18*Math.sqrt(p));d.setAttribute('class','initial-dot');$('initial').appendChild(d)}}}}update()}}
function transitions(){{$('trans').innerHTML='';if(!$('tr').checked||!F.length)return;let f=F[fr],src=cell(f.state);if(!src)return;let c0=center(...src),g=new Map();for(let e of (M.transitions[f.state]?.[f.action]||[]))g.set(e.next,(g.get(e.next)||0)+ +e.p);for(let [ns,p] of g){{if(p<1e-8)continue;let q=cell(+ns),c1=center(...q),el=document.createElementNS('http://www.w3.org/2000/svg',+ns===+f.state?'path':'line');if(+ns===+f.state)el.setAttribute('d',`M ${{c0.x}} ${{c0.y-12}} C ${{c0.x+42}} ${{c0.y-60}}, ${{c0.x+42}} ${{c0.y+60}}, ${{c0.x+2}} ${{c0.y+12}}`);else{{let dx=c1.x-c0.x,dy=c1.y-c0.y,l=Math.hypot(dx,dy)||1;el.setAttribute('x1',c0.x+dx/l*14);el.setAttribute('y1',c0.y+dy/l*14);el.setAttribute('x2',c1.x-dx/l*17);el.setAttribute('y2',c1.y-dy/l*17)}}el.setAttribute('class','arrow');el.setAttribute('stroke-width',1.2+6*p);el.setAttribute('opacity',.25+.75*p);$('trans').appendChild(el)}}}}
function current(){{$('current').innerHTML='';if(!F.length)return;let f=F[fr],c=cell(f.state);if(!c)return;let q=center(...c),d=document.createElementNS('http://www.w3.org/2000/svg','circle');d.setAttribute('cx',q.x);d.setAttribute('cy',q.y);d.setAttribute('r',15);d.setAttribute('class','current');$('current').appendChild(d);let t=document.createElementNS('http://www.w3.org/2000/svg','text');t.setAttribute('x',q.x);t.setAttribute('y',q.y+31);t.setAttribute('class','action-label');t.textContent=M.actions[f.action]??f.action;$('current').appendChild(t)}}
function update(){{let r=svg.getBoundingClientRect(),w=M.sizeY*C,h=M.sizeX*C;vp.setAttribute('transform',`translate(${{(r.width-w)/2+px}} ${{(r.height-h)/2+py}}) scale(${{zoom}})`);}}
function setFrame(i){{fr=F.length?Math.max(0,Math.min(F.length-1,i)):0;$('timeline').value=fr;$('fl').textContent=`Frame ${{F.length?fr+1:0}} / ${{F.length}}`;if(F.length){{let f=F[fr];$('step').innerHTML=`<b>State:</b> ${{f.state}}<br><b>Action:</b> ${{M.actions[f.action]??f.action}}<br><b>Reward:</b> ${{Number(f.reward).toPrecision(5)}}`}}transitions();current()}}
function select(s){{let [x,y]=cell(s),rows='';for(let a=0;a<M.nA;a++){{let g=new Map();for(let e of(M.transitions[s]?.[a]||[]))g.set(e.next,(g.get(e.next)||0)+ +e.p);rows+=`<tr><td>${{M.actions[a]??a}}</td><td>${{(+M.rewards[s][a]).toFixed(3)}}</td><td>${{[...g].filter(q=>q[1]>1e-8).map(q=>q[0]+' ('+q[1].toFixed(2)+')').join(', ')}}</td></tr>`}}$('cellinfo').innerHTML=`<b>State:</b> ${{s}}<br><b>Position:</b> (${{x}}, ${{y}})<br><b>Type:</b> ${{M.goals.includes(s)?'Goal':'Free'}}<table><tr><th>Action</th><th>Reward</th><th>Transitions</th></tr>${{rows}}</table>`}}
$('first').onclick=()=>setFrame(0);$('prev').onclick=()=>setFrame(fr-1);$('next').onclick=()=>setFrame(fr+1);$('last').onclick=()=>setFrame(F.length-1);$('timeline').max=Math.max(0,F.length-1);$('timeline').oninput=e=>setFrame(+e.target.value);
function stop(){{playing=false;if(timer)clearTimeout(timer);timer=null;$('play').textContent='▶'}}function tick(){{if(!playing)return;if(fr>=F.length-1)return stop();setFrame(fr+1);timer=setTimeout(tick,+$('speed').value)}}$('play').onclick=()=>{{if(!F.length)return;if(playing)stop();else{{playing=true;$('play').textContent='❚❚';tick()}}}};
$('tr').onchange=transitions;$('ids').onchange=draw;$('ini').onchange=draw;$('zi').onclick=()=>{{zoom=Math.min(5,zoom*1.25);update()}};$('zo').onclick=()=>{{zoom=Math.max(.25,zoom*.8);update()}};$('reset').onclick=()=>{{zoom=1;px=py=0;update()}};
let drag=null;svg.addEventListener('pointerdown',e=>{{if(e.target.closest?.('.grid-cell'))return;drag={{id:e.pointerId,x:e.clientX,y:e.clientY,px,py}};svg.setPointerCapture(e.pointerId);svg.classList.add('dragging')}});svg.addEventListener('pointermove',e=>{{if(!drag||e.pointerId!==drag.id)return;px=drag.px+e.clientX-drag.x;py=drag.py+e.clientY-drag.y;update()}});function end(e){{if(!drag||e.pointerId!==drag.id)return;try{{svg.releasePointerCapture(e.pointerId)}}catch(_){{}}drag=null;svg.classList.remove('dragging')}}svg.addEventListener('pointerup',end);svg.addEventListener('pointercancel',end);svg.addEventListener('wheel',e=>{{e.preventDefault();let r=svg.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top,old=zoom,f=e.deltaY<0?1.12:.89,nz=Math.max(.25,Math.min(5,zoom*f)),bx=(r.width-M.sizeY*C)/2,by=(r.height-M.sizeX*C)/2,wx=(mx-bx-px)/old,wy=(my-by-py)/old;zoom=nz;px=mx-bx-wx*nz;py=my-by-wy*nz;update()}},{{passive:false}});window.onresize=update;draw();setFrame(0);if({str(self.autoplay).lower()}&&F.length>1)$('play').click();
</script></body></html>'''


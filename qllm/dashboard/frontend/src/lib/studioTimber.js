import * as T from "three";

const TAU = Math.PI * 2;
export const TIMBER_SIZE = 256;
export const TIMBER_SCALE = Object.freeze([1.35, .217]);

// Smooth periodic value noise supplies irregular spacing, never frame-time RNG.
function timberNoise(u,v,columns,rows) {
  const x=u*columns,y=v*rows,ix=Math.floor(x),iy=Math.floor(y);
  const fade=t=>t*t*t*(t*(t*6-15)+10);
  const hash=(a,b)=>{
    a=((a%columns)+columns)%columns; b=((b%rows)+rows)%rows;
    let n=Math.imul(a+17,374761393)^Math.imul(b+43,668265263);
    n=Math.imul(n^(n>>>13),1274126177);
    return ((n^(n>>>16))>>>0)/4294967295;
  };
  const sx=fade(x-ix),sy=fade(y-iy);
  const a=T.MathUtils.lerp(hash(ix,iy),hash(ix+1,iy),sx);
  const b=T.MathUtils.lerp(hash(ix,iy+1),hash(ix+1,iy+1),sx);
  return T.MathUtils.lerp(a,b,sy);
}

// Periodic authored grain, not a botanical model. u runs along the timber.
// One structural cause supplies reflectance, shallow height and roughness.
export function timberField(u, v) {
  const broad=timberNoise(u,v,3,11),fine=timberNoise(u,v,7,9);
  const warp=1.4*(timberNoise(u,v,3,5)-.5)+.4*(fine-.5);
  const rings=(.5+.5*Math.sin(TAU*(14*v+warp))) ** 10;
  const pores=(.5+.5*Math.sin(TAU*(37*v+warp*.6))) ** 12;
  return Math.max(0,Math.min(1,.25+.65*broad-.28*rings*(.3+.7*fine)-.05*pores));
}

export function createTimberTextures() {
  const color = new Uint8Array(TIMBER_SIZE*TIMBER_SIZE*4);
  const surface = new Uint8Array(color.length);
  for (let y=0;y<TIMBER_SIZE;y++) for (let x=0;x<TIMBER_SIZE;x++) {
    const grain=timberField((x+.5)/TIMBER_SIZE,(y+.5)/TIMBER_SIZE);
    const i=(y*TIMBER_SIZE+x)*4, reflectance=Math.round(255*(.66+.28*grain));
    color.set([reflectance,reflectance,reflectance,255],i);
    surface.set([Math.round(255*grain),Math.round(255*(1-.17*grain)),0,255],i);
  }
  const texture=(data,name,colorSpace)=>{
    const map=new T.DataTexture(data,TIMBER_SIZE,TIMBER_SIZE,T.RGBAFormat,T.UnsignedByteType);
    map.name=name;
    map.colorSpace=colorSpace;
    map.wrapS=map.wrapT=T.RepeatWrapping;
    map.magFilter=T.LinearFilter;
    map.minFilter=T.LinearMipmapLinearFilter;
    map.generateMipmaps=true;
    map.anisotropy=4;
    map.needsUpdate=true;
    return map;
  };
  return {
    color:texture(color,"Timber linear reflectance",T.LinearSRGBColorSpace),
    surface:texture(surface,"Timber height R / roughness G",T.NoColorSpace),
  };
}

export function timberMaterial(color, maps, roughness=.8) {
  return new T.MeshStandardMaterial({color,metalness:0,roughness,
    map:maps.color,roughnessMap:maps.surface,bumpMap:maps.surface,bumpScale:.0004});
}

// Set local UVs before translating/rotating/merging. Stable member offsets avoid
// synchronized grain across neighbouring boards; lengths retain the same scale.
export function timberUV(geometry, axis="x", id="table") {
  let seed=2166136261;
  for (const char of id) seed=Math.imul(seed^char.charCodeAt(0),16777619)>>>0;
  const offsetU=(seed&65535)/65536, offsetV=(seed>>>16)/65536;
  const {position,normal,uv}=geometry.attributes;
  for(let i=0;i<position.count;i++) {
    const x=position.getX(i),y=position.getY(i),z=position.getZ(i);
    const nx=Math.abs(normal.getX(i)),ny=Math.abs(normal.getY(i)),nz=Math.abs(normal.getZ(i));
    let along,across;
    if(axis==="table") {
      // Close the veneer around the cylinder with a whole number of periods.
      // The authored .47-radius table uses two repeats (~9.4% scale adjustment).
      const repeats=Math.max(1,Math.round(TAU*Math.hypot(x,z)/TIMBER_SCALE[0]));
      along=ny>.9?x:uv.getX(i)*repeats*TIMBER_SCALE[0];
      across=ny>.9?z:y;
    } else if(axis==="x") {
      along=nx>.9?z:x; across=nx>.9?y:ny>=nz?z:y;
    } else if(axis==="z") {
      along=nz>.9?x:z; across=nz>.9?y:ny>=nx?x:y;
    } else {
      along=ny>.9?x:y; across=ny>.9?z:nz>=nx?x:z;
    }
    uv.setXY(i,along/TIMBER_SCALE[0]+offsetU,across/TIMBER_SCALE[1]+offsetV);
  }
  uv.needsUpdate=true;
  return geometry;
}

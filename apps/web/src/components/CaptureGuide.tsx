import type { ExpectedView } from "../lib/api";

export const captureViewLabels: Record<ExpectedView, string> = {
  left_front: "左前方", right_front: "右前方", left_rear: "左後方", right_rear: "右後方",
  interior_front: "前座車內", interior_rear: "後座車內", damage_detail: "車損特寫",
};

export function isExteriorView(view: ExpectedView) {
  return ["left_front", "right_front", "left_rear", "right_rear"].includes(view);
}

// Simplified outlines from the supplied reference photos, without backgrounds or plate text.
const outlines = {
  left_front: {
    body: "M112 270 Q130 185 205 160 L445 155 Q490 164 555 224 L645 294 Q827 327 878 417 L908 516 L901 567 Q842 613 590 665 L344 637 Q321 654 295 624 L261 556 L196 414 L125 359 Z",
    windows: "M229 176 L441 174 Q479 182 538 233 L619 290 L253 311 Z M209 183 L215 301 L150 274 Q158 214 209 183 Z",
    panels: "M253 322 Q455 357 611 465 L873 419 M611 465 L560 618 M151 312 L277 455 M590 636 L889 544",
    wheels: "M278 476 Q258 485 278 565 Q296 633 327 624 Q354 614 331 540 Q310 472 278 476 Z",
    plate: "M714 524 L831 485 L834 543 L716 582 Z",
  },
  right_front: {
    body: "M107 427 Q129 345 240 285 L367 178 Q418 147 557 145 L698 163 Q761 187 790 267 L815 352 L758 521 Q741 595 691 598 L650 645 Q507 682 365 666 L163 626 Q102 598 94 546 Z",
    windows: "M254 291 L381 188 Q418 173 555 172 L669 186 L651 312 Z M697 193 Q748 210 764 276 L753 345 L686 375 Z",
    panels: "M132 416 Q330 477 587 441 L649 319 M118 517 Q326 491 553 560 M766 361 L684 504 M161 611 Q405 670 635 621",
    wheels: "M702 450 Q665 447 649 528 Q631 607 662 610 Q699 617 725 535 Q747 466 702 450 Z",
    plate: "M224 534 L371 553 L373 604 L224 581 Z",
  },
  left_rear: {
    body: "M99 319 Q105 253 196 192 Q243 166 350 169 L653 195 Q729 204 765 273 L831 391 L846 526 Q834 560 758 590 L427 670 Q345 699 311 617 L199 459 L118 393 Z",
    windows: "M223 195 L341 194 L405 322 L230 290 Z M438 225 Q534 199 689 227 Q738 240 788 354 L435 386 Z M201 204 L212 289 L131 273 Z",
    panels: "M431 390 L451 537 Q466 580 537 568 L816 502 M116 340 L393 440 M419 412 L449 457 L572 442 M800 376 L821 419 M441 644 L823 549",
    wheels: "M326 460 Q293 458 307 547 Q324 649 367 647 Q409 643 385 548 Q364 465 326 460 Z",
    plate: "M651 433 L754 411 L758 466 L656 492 Z",
  },
  right_rear: {
    body: "M173 211 Q211 169 389 166 L681 181 Q767 201 820 268 L880 341 L858 416 L768 551 Q733 638 686 629 L602 657 Q421 647 178 552 Q106 519 112 444 L128 307 Z",
    windows: "M190 222 Q272 190 400 213 Q455 220 479 273 L473 382 L141 319 Z M524 260 Q559 218 670 214 L669 340 L500 387 Z M694 219 Q755 236 791 294 L702 332 Z",
    panels: "M141 334 L420 450 Q414 536 345 538 L153 458 M446 396 L466 438 L579 451 M856 354 L505 467 M144 508 Q353 611 597 628",
    wheels: "M709 471 Q669 463 637 546 Q602 634 640 644 Q684 654 725 567 Q763 484 709 471 Z",
    plate: "M203 379 L321 410 L316 465 L198 427 Z",
  },
};

// Traced from the admin's 600 × 800 interior reference photos: viewed through
// the open passenger door, rather than looking straight at two upright seats.
const interiorOutlines = {
  interior_front: {
    body: "M62 293 Q71 276 91 300 L164 420 Q207 439 270 434 Q310 432 330 459 L366 549 Q380 597 344 624 L207 654 Q175 653 157 624 L111 492 Z M58 186 Q71 164 91 180 L178 301 Q237 306 287 326 Q312 342 329 382 Q302 409 223 417 L175 406 L103 283 Z",
    detail: "M159 491 Q228 449 308 455 M178 602 L344 577 M214 474 L239 588 M260 463 L291 580 M181 353 L296 334 M230 345 L246 394 M73 127 Q278 110 503 229 L588 282 M416 220 L490 248 L452 330 L392 300 Z M391 305 L445 340 L403 383 L356 361 M368 371 L341 417 L300 445 L416 420 M389 452 L559 458 L579 637 L422 660 Z",
    steering: "M367 188 C395 175 391 234 366 276 C339 318 304 304 318 264 C329 223 346 195 367 188 Z M326 253 L374 237 M351 244 L342 289",
  },
  interior_rear: {
    body: "M41 325 Q57 306 79 338 L151 418 Q202 399 245 405 L290 435 Q337 493 368 560 Q378 599 345 645 L309 681 Q240 680 182 620 L94 468 Z M398 215 Q424 195 458 215 L476 283 Q471 310 449 337 Q506 354 510 420 L499 548 Q498 568 477 578 L473 639 L413 640 L353 486 Q314 419 321 354 Q324 322 350 313 Z",
    detail: "M58 389 L177 534 Q224 551 343 523 M152 423 L252 413 M165 467 L287 443 M186 516 L317 483 M178 594 Q251 613 361 567 M225 459 L263 501 M279 490 L313 528 M310 276 L349 291 Q382 304 395 351 M318 214 Q295 203 295 230 L286 266 Q290 285 319 280 L340 274 L348 214 Z M359 332 Q414 316 449 337 M404 351 L471 570 M391 553 L403 594 M316 581 Q358 631 382 680 L452 663 M85 224 Q226 206 287 214 M82 237 L127 287 L292 284",
    steering: null,
  },
};

export function CaptureGuide({ view, compact = false }: { view: ExpectedView; compact?: boolean }) {
  if (view === "damage_detail") return null;
  const exterior = isExteriorView(view) ? outlines[view as keyof typeof outlines] : null;
  const interior = view === "interior_front" || view === "interior_rear" ? interiorOutlines[view] : null;
  return <svg className={`capture-guide ${compact ? "capture-guide--compact" : ""}`}
    viewBox={interior ? "0 0 600 800" : "0 0 1000 750"} role="img" aria-label={`${captureViewLabels[view]}半透明拍攝對齊框`}>
    {exterior ? <>
      <path className="capture-guide__body" d={exterior.body} />
      <path className="capture-guide__detail" d={exterior.windows} />
      <path className="capture-guide__detail" d={exterior.panels} />
      <path className="capture-guide__detail" d={exterior.wheels} />
      <path className="capture-guide__plate" d={exterior.plate} />
    </> : interior && <>
      <path className="capture-guide__body" d={interior.body} />
      <path className="capture-guide__detail" d={interior.detail} />
      {interior.steering && <path className="capture-guide__detail" d={interior.steering} />}
    </>}
  </svg>;
}

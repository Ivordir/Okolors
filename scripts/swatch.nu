#!/bin/nu --stdin

let w = 40

def main [] {
  let colors = ($in | lines | each { split row ' ' })

  let inner = (
    $colors
    | enumerate
    | each { |line|
      if $line.index == 0 {
        $line.item
        | enumerate
        | each { |color|
          $'  <use href="#swatch" x="($color.index * $w)" fill="rgb($color.item)"/>'
        }
      } else {
        let y = ($line.index * $w)
        $line.item
        | enumerate
        | each { |color|
          let x = ($color.index * $w)
          $'  <use href="#swatch" x="($x)" y="($y)" fill="rgb($color.item)"/>'
        }
      }
    }
    | flatten
    | str join (char newline)
  )

  let width = ($colors.0 | length) * $w
  let height = ($colors | length) * $w

  print $'<svg height="($height)" viewBox="0 0 ($width) ($height)" xmlns="http://www.w3.org/2000/svg">
  <rect id="swatch" width="($w)" height="($w)"/>
($inner)
</svg>'
}

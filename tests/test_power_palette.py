# 电源图例契约独立核对：按说明书的资源语义配色，灰度打印仍有纹理区分。
import unittest


class PowerPaletteTests(unittest.TestCase):
    def test_semantic_colors_hatches_and_fresh_results(self):
        import zyo_power.uc_plot as plot
        self.assertTrue(hasattr(plot,'stack_style'))
        resources=['thermal','wind','solar','hydro','storage','ens']
        expected=['#ff0000','#00cc00','#ffff00','#00cccc','#0000ff','#dddddd']
        style=plot.stack_style(resources)
        self.assertEqual(style['colors'],expected)
        self.assertEqual(len(set(style['hatches'])),len(resources))
        style['colors'][0]='changed'
        self.assertEqual(plot.stack_style(resources)['colors'],expected)
        with self.assertRaises(ValueError): plot.stack_style(['unknown_resource'])


if __name__=='__main__': unittest.main()

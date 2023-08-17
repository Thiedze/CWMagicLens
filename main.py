from magic_service import MagicService

if __name__ == '__main__':
    # din a5 width in pixel (30dpi) => 248
    # din a5 height in pixel (30dpi) =>	175

    magic_service = MagicService()
    magic_service.do_magic_with_path("input/sewing_machine.jpg", "input/master.jpg", (248, 175))
